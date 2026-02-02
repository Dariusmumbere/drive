# main.py
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional, List
from jose import JWTError, jwt
from passlib.context import CryptContext
import uuid
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import os
import mimetypes
import io
import logging
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backblaze B2 Configuration
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "your-bucket-name")
B2_ENDPOINT_URL = os.getenv("B2_ENDPOINT_URL", "https://s3.us-east-005.backblazeb2.com")
B2_KEY_ID = os.getenv("B2_KEY_ID", "your-key-id")
B2_APPLICATION_KEY = os.getenv("B2_APPLICATION_KEY", "your-application-key")

# Initialize B2 client
b2_client = boto3.client(
    's3',
    endpoint_url=B2_ENDPOINT_URL,
    aws_access_key_id=B2_KEY_ID,
    aws_secret_access_key=B2_APPLICATION_KEY
)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://drive_ckby_user:BEllpEiHkxMdRTnwCx76KJhSDABSYBuN@dpg-d60bhe4hg0os73a7u4d0-a/drive_ckby")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    storage_used = Column(Integer, default=0)  # in bytes
    storage_limit = Column(Integer, default=10737418240)  # 10GB default
    created_at = Column(DateTime, default=datetime.utcnow)
    
    files = relationship("File", back_populates="owner")
    folders = relationship("Folder", back_populates="owner")

class File(Base):
    __tablename__ = "files"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    original_filename = Column(String)
    file_size = Column(Integer)
    mime_type = Column(String)
    b2_filename = Column(String, unique=True)  # Path in Backblaze
    folder_id = Column(Integer, ForeignKey("folders.id"), nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    owner = relationship("User", back_populates="files")
    folder = relationship("Folder", back_populates="files")

class Folder(Base):
    __tablename__ = "folders"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    parent_id = Column(Integer, ForeignKey("folders.id"), nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    owner = relationship("User", back_populates="folders")
    files = relationship("File", back_populates="folder")
    subfolders = relationship("Folder", backref="parent", remote_side=[id])

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models
class UserBase(BaseModel):
    email: EmailStr
    full_name: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool
    storage_used: int
    storage_limit: int
    created_at: datetime
    
    class Config:
        from_attributes = True  # Changed from orm_mode

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class FileBase(BaseModel):
    filename: str
    file_size: int
    mime_type: str

class FileCreate(FileBase):
    folder_id: Optional[int] = None
    is_public: bool = False

class FileResponse(FileBase):
    id: int
    original_filename: str
    b2_filename: str
    folder_id: Optional[int]
    owner_id: int
    is_public: bool
    created_at: datetime
    updated_at: datetime
    download_url: Optional[str] = None
    preview_url: Optional[str] = None
    
    class Config:
        from_attributes = True  # Changed from orm_mode

class FolderBase(BaseModel):
    name: str
    parent_id: Optional[int] = None

class FolderCreate(FolderBase):
    pass

class FolderResponse(FolderBase):
    id: int
    owner_id: int
    created_at: datetime
    updated_at: datetime
    file_count: int = 0
    folder_count: int = 0
    
    class Config:
        from_attributes = True  # Changed from orm_mode

class StorageInfo(BaseModel):
    used: int
    limit: int
    percentage: float
    files_count: int
    folders_count: int

# Auth setup
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title="Cloud Drive API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _prehash_password(password: str) -> str:
    # Convert to bytes and hash with SHA-256
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def get_password_hash(password: str) -> str:
    return pwd_context.hash(_prehash_password(password))

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(
        _prehash_password(plain_password),
        hashed_password
    )

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    
    user = get_user_by_email(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    return user

# B2 Helper Functions
async def upload_to_b2(file: UploadFile, user_id: int) -> str:
    """Upload a file to Backblaze B2 and return the B2 filename"""
    try:
        # Generate unique filename with user folder
        file_extension = Path(file.filename).suffix if file.filename else ''
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        b2_filename = f"users/{user_id}/{unique_filename}"
        
        # Read file content
        file_content = await file.read()
        
        # Upload to B2
        b2_client.put_object(
            Bucket=B2_BUCKET_NAME,
            Key=b2_filename,
            Body=file_content,
            ContentType=file.content_type
        )
        
        logger.info(f"File uploaded to B2: {b2_filename}")
        return b2_filename
        
    except Exception as e:
        logger.error(f"Error uploading file to B2: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

async def delete_from_b2(b2_filename: str):
    """Delete a file from Backblaze B2"""
    try:
        b2_client.delete_object(Bucket=B2_BUCKET_NAME, Key=b2_filename)
        logger.info(f"File deleted from B2: {b2_filename}")
    except Exception as e:
        logger.error(f"Error deleting file from B2: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

async def generate_presigned_url(b2_filename: str, expiration: int = 3600):
    """Generate a presigned URL for private B2 objects"""
    try:
        if not b2_filename:
            return None
        
        url = b2_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': B2_BUCKET_NAME,
                'Key': b2_filename
            },
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        logger.error(f"Error generating presigned URL: {e}")
        return None

def check_storage_limit(db: Session, user: User, additional_size: int) -> bool:
    """Check if user has enough storage space"""
    return user.storage_used + additional_size <= user.storage_limit

def update_storage_used(db: Session, user: User, file_size: int, operation: str = "add"):
    """Update user's storage usage"""
    if operation == "add":
        user.storage_used += file_size
    elif operation == "subtract":
        user.storage_used -= file_size
        if user.storage_used < 0:
            user.storage_used = 0
    db.commit()

# Auth Endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email, 
        hashed_password=hashed_password, 
        full_name=user.full_name
    )
    db.add(db_user)
    
    # Create root folder for user
    root_folder = Folder(
        name="Root",
        owner_id=db_user.id
    )
    db.add(root_folder)
    
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/me/", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# File Management Endpoints
@app.post("/files/upload", response_model=FileResponse)
async def upload_file(
    file: UploadFile = File(),  # FIXED: Removed the ... inside File()
    folder_id: Optional[int] = None,
    is_public: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a file to cloud storage"""
    
    # Check storage limit
    file_size = 0
    file_content = await file.read()
    file_size = len(file_content)
    
    if not check_storage_limit(db, current_user, file_size):
        raise HTTPException(
            status_code=400,
            detail="Storage limit exceeded. Please upgrade your plan or delete some files."
        )
    
    # Reset file pointer
    file.file = io.BytesIO(file_content)
    
    # Upload to Backblaze B2
    b2_filename = await upload_to_b2(file, current_user.id)
    
    # Create file record in database
    db_file = File(
        filename=Path(file.filename).stem if file.filename else f"file_{uuid.uuid4().hex[:8]}",
        original_filename=file.filename or "unnamed_file",
        file_size=file_size,
        mime_type=file.content_type or mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream",
        b2_filename=b2_filename,
        folder_id=folder_id,
        owner_id=current_user.id,
        is_public=is_public
    )
    
    db.add(db_file)
    
    # Update user storage usage
    update_storage_used(db, current_user, file_size, "add")
    
    db.commit()
    db.refresh(db_file)
    
    # Generate download URL
    download_url = await generate_presigned_url(b2_filename)
    
    # Create response using dict instead of from_orm for better compatibility
    response_data = {
        "id": db_file.id,
        "filename": db_file.filename,
        "original_filename": db_file.original_filename,
        "file_size": db_file.file_size,
        "mime_type": db_file.mime_type,
        "b2_filename": db_file.b2_filename,
        "folder_id": db_file.folder_id,
        "owner_id": db_file.owner_id,
        "is_public": db_file.is_public,
        "created_at": db_file.created_at,
        "updated_at": db_file.updated_at,
        "download_url": download_url
    }
    
    # Add preview URL for images and PDFs
    if db_file.mime_type and db_file.mime_type.startswith(('image/', 'application/pdf')):
        response_data["preview_url"] = download_url
    else:
        response_data["preview_url"] = None
    
    return FileResponse(**response_data)

@app.get("/files/{file_id}", response_model=FileResponse)
async def get_file(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get file information and download URL"""
    db_file = db.query(File).filter(
        File.id == file_id,
        File.owner_id == current_user.id
    ).first()
    
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Generate download URL
    download_url = await generate_presigned_url(db_file.b2_filename)
    
    # Create response
    response_data = {
        "id": db_file.id,
        "filename": db_file.filename,
        "original_filename": db_file.original_filename,
        "file_size": db_file.file_size,
        "mime_type": db_file.mime_type,
        "b2_filename": db_file.b2_filename,
        "folder_id": db_file.folder_id,
        "owner_id": db_file.owner_id,
        "is_public": db_file.is_public,
        "created_at": db_file.created_at,
        "updated_at": db_file.updated_at,
        "download_url": download_url
    }
    
    # Add preview URL for images and PDFs
    if db_file.mime_type and db_file.mime_type.startswith(('image/', 'application/pdf')):
        response_data["preview_url"] = download_url
    else:
        response_data["preview_url"] = None
    
    return FileResponse(**response_data)

@app.get("/files/{file_id}/download")
async def download_file(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Download a file"""
    db_file = db.query(File).filter(
        File.id == file_id,
        File.owner_id == current_user.id
    ).first()
    
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Generate presigned URL for direct download
    download_url = await generate_presigned_url(db_file.b2_filename)
    
    return {"download_url": download_url, "filename": db_file.original_filename}

@app.delete("/files/{file_id}")
async def delete_file(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a file from cloud storage"""
    db_file = db.query(File).filter(
        File.id == file_id,
        File.owner_id == current_user.id
    ).first()
    
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Delete from Backblaze B2
    await delete_from_b2(db_file.b2_filename)
    
    # Update user storage usage
    update_storage_used(db, current_user, db_file.file_size, "subtract")
    
    # Delete from database
    db.delete(db_file)
    db.commit()
    
    return {"message": "File deleted successfully"}

@app.get("/files", response_model=List[FileResponse])
async def list_files(
    folder_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List files in a folder or root"""
    query = db.query(File).filter(File.owner_id == current_user.id)
    
    if folder_id is not None:
        query = query.filter(File.folder_id == folder_id)
    else:
        query = query.filter(File.folder_id.is_(None))
    
    files = query.offset(skip).limit(limit).all()
    
    # Generate response
    response_files = []
    for file in files:
        download_url = await generate_presigned_url(file.b2_filename)
        
        response_data = {
            "id": file.id,
            "filename": file.filename,
            "original_filename": file.original_filename,
            "file_size": file.file_size,
            "mime_type": file.mime_type,
            "b2_filename": file.b2_filename,
            "folder_id": file.folder_id,
            "owner_id": file.owner_id,
            "is_public": file.is_public,
            "created_at": file.created_at,
            "updated_at": file.updated_at,
            "download_url": download_url
        }
        
        if file.mime_type and file.mime_type.startswith(('image/', 'application/pdf')):
            response_data["preview_url"] = download_url
        else:
            response_data["preview_url"] = None
        
        response_files.append(FileResponse(**response_data))
    
    return response_files

# Folder Management Endpoints
@app.post("/folders/", response_model=FolderResponse)
async def create_folder(
    folder: FolderCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new folder"""
    # Check if parent folder exists and belongs to user
    if folder.parent_id:
        parent_folder = db.query(Folder).filter(
            Folder.id == folder.parent_id,
            Folder.owner_id == current_user.id
        ).first()
        if not parent_folder:
            raise HTTPException(status_code=404, detail="Parent folder not found")
    
    # Check if folder with same name already exists in same location
    existing_folder = db.query(Folder).filter(
        Folder.name == folder.name,
        Folder.parent_id == folder.parent_id,
        Folder.owner_id == current_user.id
    ).first()
    
    if existing_folder:
        raise HTTPException(status_code=400, detail="Folder with this name already exists")
    
    db_folder = Folder(
        name=folder.name,
        parent_id=folder.parent_id,
        owner_id=current_user.id
    )
    
    db.add(db_folder)
    db.commit()
    db.refresh(db_folder)
    
    # Get file and folder counts
    file_count = db.query(File).filter(
        File.folder_id == db_folder.id,
        File.owner_id == current_user.id
    ).count()
    
    folder_count = db.query(Folder).filter(
        Folder.parent_id == db_folder.id,
        Folder.owner_id == current_user.id
    ).count()
    
    response_data = {
        "id": db_folder.id,
        "name": db_folder.name,
        "parent_id": db_folder.parent_id,
        "owner_id": db_folder.owner_id,
        "created_at": db_folder.created_at,
        "updated_at": db_folder.updated_at,
        "file_count": file_count,
        "folder_count": folder_count
    }
    
    return FolderResponse(**response_data)

@app.get("/folders/{folder_id}", response_model=FolderResponse)
async def get_folder(
    folder_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get folder information"""
    db_folder = db.query(Folder).filter(
        Folder.id == folder_id,
        Folder.owner_id == current_user.id
    ).first()
    
    if not db_folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    
    # Get file and folder counts
    file_count = db.query(File).filter(
        File.folder_id == folder_id,
        File.owner_id == current_user.id
    ).count()
    
    folder_count = db.query(Folder).filter(
        Folder.parent_id == folder_id,
        Folder.owner_id == current_user.id
    ).count()
    
    response_data = {
        "id": db_folder.id,
        "name": db_folder.name,
        "parent_id": db_folder.parent_id,
        "owner_id": db_folder.owner_id,
        "created_at": db_folder.created_at,
        "updated_at": db_folder.updated_at,
        "file_count": file_count,
        "folder_count": folder_count
    }
    
    return FolderResponse(**response_data)

@app.get("/folders", response_model=List[FolderResponse])
async def list_folders(
    parent_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List folders"""
    query = db.query(Folder).filter(Folder.owner_id == current_user.id)
    
    if parent_id is not None:
        query = query.filter(Folder.parent_id == parent_id)
    else:
        query = query.filter(Folder.parent_id.is_(None))
    
    folders = query.offset(skip).limit(limit).all()
    
    # Get counts for each folder
    response_folders = []
    for folder in folders:
        file_count = db.query(File).filter(
            File.folder_id == folder.id,
            File.owner_id == current_user.id
        ).count()
        
        folder_count = db.query(Folder).filter(
            Folder.parent_id == folder.id,
            Folder.owner_id == current_user.id
        ).count()
        
        response_data = {
            "id": folder.id,
            "name": folder.name,
            "parent_id": folder.parent_id,
            "owner_id": folder.owner_id,
            "created_at": folder.created_at,
            "updated_at": folder.updated_at,
            "file_count": file_count,
            "folder_count": folder_count
        }
        
        response_folders.append(FolderResponse(**response_data))
    
    return response_folders

@app.delete("/folders/{folder_id}")
async def delete_folder(
    folder_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a folder and all its contents"""
    db_folder = db.query(Folder).filter(
        Folder.id == folder_id,
        Folder.owner_id == current_user.id
    ).first()
    
    if not db_folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    
    # Get all files in this folder and subfolders
    def get_all_files_in_folder(folder_id):
        files = []
        # Get files in current folder
        folder_files = db.query(File).filter(
            File.folder_id == folder_id,
            File.owner_id == current_user.id
        ).all()
        files.extend(folder_files)
        
        # Get subfolders
        subfolders = db.query(Folder).filter(
            Folder.parent_id == folder_id,
            Folder.owner_id == current_user.id
        ).all()
        
        for subfolder in subfolders:
            files.extend(get_all_files_in_folder(subfolder.id))
        
        return files
    
    all_files = get_all_files_in_folder(folder_id)
    
    # Delete all files from Backblaze B2 and update storage
    total_deleted_size = 0
    for file in all_files:
        try:
            await delete_from_b2(file.b2_filename)
            total_deleted_size += file.file_size
        except Exception as e:
            logger.error(f"Error deleting file {file.id} from B2: {e}")
    
    # Update user storage usage
    if total_deleted_size > 0:
        update_storage_used(db, current_user, total_deleted_size, "subtract")
    
    # Delete all files and folders from database
    # (In production, you might want to use cascade delete or recursive CTE)
    for file in all_files:
        db.delete(file)
    
    # Delete all subfolders recursively
    def delete_subfolders(parent_id):
        subfolders = db.query(Folder).filter(
            Folder.parent_id == parent_id,
            Folder.owner_id == current_user.id
        ).all()
        
        for subfolder in subfolders:
            delete_subfolders(subfolder.id)
            db.delete(subfolder)
    
    delete_subfolders(folder_id)
    
    # Delete the main folder
    db.delete(db_folder)
    db.commit()
    
    return {"message": "Folder and all contents deleted successfully"}

# Storage Information
@app.get("/storage/info", response_model=StorageInfo)
async def get_storage_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get storage usage information"""
    files_count = db.query(File).filter(File.owner_id == current_user.id).count()
    folders_count = db.query(Folder).filter(Folder.owner_id == current_user.id).count()
    
    percentage = (current_user.storage_used / current_user.storage_limit * 100) if current_user.storage_limit > 0 else 0
    
    return StorageInfo(
        used=current_user.storage_used,
        limit=current_user.storage_limit,
        percentage=round(percentage, 2),
        files_count=files_count,
        folders_count=folders_count
    )

# Search Endpoints
@app.get("/search")
async def search_files(
    query: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Search files by name"""
    files = db.query(File).filter(
        File.owner_id == current_user.id,
        (File.filename.ilike(f"%{query}%")) | (File.original_filename.ilike(f"%{query}%"))
    ).limit(50).all()
    
    response_files = []
    for file in files:
        download_url = await generate_presigned_url(file.b2_filename)
        
        response_data = {
            "id": file.id,
            "filename": file.filename,
            "original_filename": file.original_filename,
            "file_size": file.file_size,
            "mime_type": file.mime_type,
            "b2_filename": file.b2_filename,
            "folder_id": file.folder_id,
            "owner_id": file.owner_id,
            "is_public": file.is_public,
            "created_at": file.created_at,
            "updated_at": file.updated_at,
            "download_url": download_url
        }
        
        if file.mime_type and file.mime_type.startswith(('image/', 'application/pdf')):
            response_data["preview_url"] = download_url
        else:
            response_data["preview_url"] = None
        
        response_files.append(FileResponse(**response_data))
    
    return response_files

# Public File Access (for shared files)
@app.get("/public/files/{file_id}")
async def get_public_file(
    file_id: int,
    db: Session = Depends(get_db)
):
    """Get a public file (no authentication required)"""
    db_file = db.query(File).filter(
        File.id == file_id,
        File.is_public == True
    ).first()
    
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found or not public")
    
    # Generate download URL
    download_url = await generate_presigned_url(db_file.b2_filename)
    
    response_data = {
        "id": db_file.id,
        "filename": db_file.filename,
        "original_filename": db_file.original_filename,
        "file_size": db_file.file_size,
        "mime_type": db_file.mime_type,
        "b2_filename": db_file.b2_filename,
        "folder_id": db_file.folder_id,
        "owner_id": db_file.owner_id,
        "is_public": db_file.is_public,
        "created_at": db_file.created_at,
        "updated_at": db_file.updated_at,
        "download_url": download_url
    }
    
    if db_file.mime_type and db_file.mime_type.startswith(('image/', 'application/pdf')):
        response_data["preview_url"] = download_url
    else:
        response_data["preview_url"] = None
    
    return FileResponse(**response_data)

@app.put("/files/{file_id}/share")
async def toggle_file_share(
    file_id: int,
    is_public: bool,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Toggle file sharing (make public/private)"""
    db_file = db.query(File).filter(
        File.id == file_id,
        File.owner_id == current_user.id
    ).first()
    
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")
    
    db_file.is_public = is_public
    db.commit()
    db.refresh(db_file)
    
    return {"message": f"File is now {'public' if is_public else 'private'}", "is_public": is_public}

# Health check
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Root endpoint
@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Cloud Drive API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "auth": ["POST /token", "POST /users/", "GET /users/me/"],
            "files": ["POST /files/upload", "GET /files", "GET /files/{id}", "DELETE /files/{id}"],
            "folders": ["POST /folders/", "GET /folders", "GET /folders/{id}", "DELETE /folders/{id}"],
            "storage": ["GET /storage/info"],
            "search": ["GET /search"],
            "sharing": ["GET /public/files/{id}", "PUT /files/{id}/share"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Ensure database tables are created
    Base.metadata.create_all(bind=engine)
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
