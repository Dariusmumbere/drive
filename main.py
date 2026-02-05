# main.py
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File as FastAPIFile, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Text, BigInteger, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from pydantic import BaseModel, EmailStr, validator
from datetime import datetime, timedelta
from typing import Optional, List, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
import uuid
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import os
import mimetypes
import io
import logging
import hashlib
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backblaze B2 Configuration
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "uploads-dir")
B2_ENDPOINT_URL = os.getenv("B2_ENDPOINT_URL", "https://s3.us-east-005.backblazeb2.com")
B2_KEY_ID = os.getenv("B2_KEY_ID", "0055ca7845641d30000000003")
B2_APPLICATION_KEY = os.getenv("B2_APPLICATION_KEY", "K005biwK475Ji5j7PGdbvOqcnNQDx4I")

b2_client = boto3.client(
    "s3",
    endpoint_url="https://s3.us-east-005.backblazeb2.com",
    aws_access_key_id=B2_KEY_ID,
    aws_secret_access_key=B2_APPLICATION_KEY,
    region_name="us-east-005",
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"}
    )
)

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "304984836974-0o5rdilg1pdvg3almd5js4b42je8p6e3.apps.googleusercontent.com")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "https://drive-t0my.onrender.com/auth/google/callback")

# Import Google auth libraries
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

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
    hashed_password = Column(String, nullable=True)  # Nullable for Google OAuth users
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    storage_used = Column(BigInteger, default=0)
    storage_limit = Column(BigInteger, default=10737418240)  # 10GB in bytes
    created_at = Column(DateTime, default=datetime.utcnow)
    auth_provider = Column(String, default="email")  # 'email' or 'google'
    google_id = Column(String, unique=True, nullable=True)  # Google user ID
    
    files = relationship("File", back_populates="owner", cascade="all, delete-orphan")
    folders = relationship("Folder", back_populates="owner", cascade="all, delete-orphan")

class File(Base):
    __tablename__ = "files"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    original_filename = Column(String)
    file_size = Column(BigInteger)
    mime_type = Column(String)
    b2_filename = Column(String, unique=True)  # Path in Backblaze
    folder_id = Column(Integer, ForeignKey("folders.id", ondelete="CASCADE"), nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    owner = relationship("User", back_populates="files")
    folder = relationship("Folder", back_populates="files")

class Folder(Base):
    __tablename__ = "folders"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    parent_id = Column(Integer, ForeignKey("folders.id"), nullable=True)

    # ONE → MANY (parent → subfolders)
    subfolders = relationship(
        "Folder",
        back_populates="parent",
        cascade="all, delete-orphan",
        single_parent=True
    )

    # MANY → ONE (child → parent)
    parent = relationship(
        "Folder",
        back_populates="subfolders",
        remote_side=[id]
    )
    
    # Relationship to owner
    owner = relationship("User", back_populates="folders")
    
    # Relationship to files
    files = relationship("File", back_populates="folder", cascade="all, delete-orphan")

# Create tables
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Tables created or already exist")
except Exception as e:
    logger.error(f"Could not create tables: {e}")

# Pydantic Models
class UserBase(BaseModel):
    email: EmailStr
    full_name: str

class UserCreate(UserBase):
    password: str
    
    @validator('password')
    def password_length(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        return v

class UserResponse(UserBase):
    id: int
    is_active: bool
    storage_used: int
    storage_limit: int
    created_at: datetime
    auth_provider: str
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    user: Optional[UserResponse] = None

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
        from_attributes = True

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
        from_attributes = True

class StorageInfo(BaseModel):
    used: int
    limit: int
    percentage: float
    files_count: int
    folders_count: int

class ShareRequest(BaseModel):
    is_public: bool

class GoogleUserInfo(BaseModel):
    email: str
    name: str
    sub: str  # Google user ID

class GoogleAuthRequest(BaseModel):
    credential: str  # Changed from 'token' to 'credential'

class GoogleAuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

# Auth setup
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# Use argon2 for password hashing
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title="Cloud Drive API")

# Configure CORS - Updated with more permissive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dariusmumbere.github.io",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
        "https://drive-t0my.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=[
        "Authorization", 
        "Content-Type", 
        "Accept", 
        "Origin", 
        "X-Requested-With",
        "X-CSRF-Token",
        "Access-Control-Allow-Origin",
        "Access-Control-Allow-Credentials",
        "Access-Control-Allow-Headers"
    ],
    expose_headers=["*"],
    max_age=3600,
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(password: str) -> str:
    """Hash a password for storing."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a stored password against one provided by user."""
    return pwd_context.verify(plain_password, hashed_password)

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def get_user_by_google_id(db: Session, google_id: str):
    return db.query(User).filter(User.google_id == google_id).first()

def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user:
        return False
    if not user.hashed_password:  # User created via Google OAuth
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
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user

def create_or_update_user_from_google(db: Session, google_user_info: GoogleUserInfo):
    """Create or update user from Google OAuth data"""
    # Check if user exists by Google ID
    user = get_user_by_google_id(db, google_user_info.sub)
    
    if user:
        # Update user info if needed
        if user.email != google_user_info.email or user.full_name != google_user_info.name:
            user.email = google_user_info.email
            user.full_name = google_user_info.name
            db.commit()
            db.refresh(user)
        return user
    
    # Check if user exists by email
    user = get_user_by_email(db, google_user_info.email)
    
    if user:
        # Link Google account to existing email account
        user.google_id = google_user_info.sub
        user.auth_provider = "google"
        db.commit()
        db.refresh(user)
        return user
    
    # Create new user
    user = User(
        email=google_user_info.email,
        full_name=google_user_info.name,
        google_id=google_user_info.sub,
        auth_provider="google",
        hashed_password=None  # No password for Google users
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
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
        user.storage_used = max(0, user.storage_used - file_size)
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
    return Token(access_token=access_token, token_type="bearer", user=user)

# Google OAuth Endpoints
@app.post("/auth/google", response_model=GoogleAuthResponse)
async def google_auth(payload: GoogleAuthRequest, db: Session = Depends(get_db)):
    """Authenticate with Google token"""
    credential = payload.credential
    if not credential:
        raise HTTPException(status_code=400, detail="Credential missing")

    try:
        # Verify the Google token
        info = id_token.verify_oauth2_token(
            credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        # Create Google user info
        google_user = GoogleUserInfo(
            email=info["email"],
            name=info.get("name", ""),
            sub=info["sub"]
        )
        
        # Create or update user in database
        user = create_or_update_user_from_google(db, google_user)
        
        # Create JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        
        return GoogleAuthResponse(
            access_token=access_token,
            token_type="bearer",
            user=user
        )

    except ValueError as e:
        logger.error(f"Google token validation error: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid Google token")
    except Exception as e:
        logger.error(f"Google authentication error: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication failed")

# Google OAuth Callback for redirect flow
@app.get("/auth/google/callback")
async def google_auth_callback(
    request: Request,
    db: Session = Depends(get_db)
):
    """Handle Google OAuth callback from frontend"""
    try:
        # Get token from query parameters
        token = request.query_params.get("token")
        if not token:
            return HTMLResponse(content="""
            <html>
                <body>
                    <h2>Authentication Error</h2>
                    <p>No token provided</p>
                    <script>
                        window.opener.postMessage({ error: "No token provided" }, "*");
                        window.close();
                    </script>
                </body>
            </html>
            """)
        
        # Verify the Google token
        info = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        # Create Google user info
        google_user = GoogleUserInfo(
            email=info["email"],
            name=info.get("name", ""),
            sub=info["sub"]
        )
        
        # Create or update user in database
        user = create_or_update_user_from_google(db, google_user)
        
        # Create JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        
        # Return HTML that sends message to opener and closes
        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Authentication Complete</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .success {{
                    color: #4CAF50;
                    font-size: 24px;
                    margin-bottom: 20px;
                }}
                .loading {{
                    margin-top: 20px;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success">✓ Authentication Successful!</div>
                <p>You will be redirected shortly...</p>
                <div class="loading">Loading...</div>
            </div>
            <script>
                // Send authentication data back to the opener window
                const authData = {{
                    access_token: "{access_token}",
                    token_type: "bearer",
                    user: {{
                        id: {user.id},
                        email: "{user.email}",
                        full_name: "{user.full_name}",
                        is_active: {str(user.is_active).lower()},
                        storage_used: {user.storage_used},
                        storage_limit: {user.storage_limit},
                        created_at: "{user.created_at.isoformat() if user.created_at else ''}",
                        auth_provider: "{user.auth_provider}"
                    }}
                }};
                
                window.opener.postMessage({{
                    type: 'google-auth-success',
                    data: authData
                }}, "*");
                
                // Close the popup after 1 second
                setTimeout(() => {{
                    window.close();
                }}, 1000);
            </script>
        </body>
        </html>
        """)
        
    except Exception as e:
        logger.error(f"Google callback error: {str(e)}")
        return HTMLResponse(content=f"""
        <html>
            <body>
                <h2>Authentication Error</h2>
                <p>Error: {str(e)}</p>
                <script>
                    window.opener.postMessage({{ type: 'google-auth-error', error: "{str(e)}" }}, "*");
                    window.close();
                </script>
            </body>
        </html>
        """)

# Google OAuth HTML page for direct testing
@app.get("/auth/google/page")
async def google_auth_page():
    """Generate HTML page for Google Sign-In button"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Google Sign-In</title>
        <script src="https://accounts.google.com/gsi/client" async defer></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f5f5f5;
            }}
            .container {{
                text-align: center;
                background-color: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Sign in with Google</h2>
            <p>This is a test page for Google Sign-In.</p>
            
            <div id="g_id_onload"
                data-client_id="{GOOGLE_CLIENT_ID}"
                data-context="signin"
                data-ux_mode="popup"
                data-callback="handleCredentialResponse"
                data-auto_prompt="false">
            </div>
            
            <div class="g_id_signin"
                data-type="standard"
                data-shape="rectangular"
                data-theme="outline"
                data-text="signin_with"
                data-size="large"
                data-logo_alignment="left">
            </div>
            
            <div id="result" style="margin-top: 20px;"></div>
        </div>
        
        <script>
            function handleCredentialResponse(response) {{
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<p>Authenticating...</p>';
                
                // Send the credential to backend
                fetch('/auth/google/callback?token=' + encodeURIComponent(response.credential))
                    .then(response => response.text())
                    .then(html => {{
                        document.body.innerHTML = html;
                    }})
                    .catch(error => {{
                        resultDiv.innerHTML = '<p style="color: red;">Authentication failed. Please try again.</p>';
                        console.error('Error:', error);
                    }});
            }}
            
            // Initialize Google Sign-In
            window.onload = function() {{
                google.accounts.id.initialize({{
                    client_id: '{GOOGLE_CLIENT_ID}',
                    callback: handleCredentialResponse
                }});
                google.accounts.id.renderButton(
                    document.querySelector('.g_id_signin'),
                    {{ theme: "outline", size: "large" }}
                );
            }};
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/users/", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email, 
        hashed_password=hashed_password, 
        full_name=user.full_name,
        auth_provider="email"
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/me/", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# File Management Endpoints
@app.post("/files/upload", response_model=FileResponse)
async def upload_file(
    file: UploadFile = FastAPIFile(...),
    folder_id: Optional[int] = Query(None),
    is_public: bool = Query(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a file to cloud storage"""
    
    # Check storage limit
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
    """Get download URL for a file"""
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
    folder_id: Optional[Union[int, str]] = Query(None),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List files in a folder or root"""
    query = db.query(File).filter(File.owner_id == current_user.id)
    
    # Handle folder_id parameter (can be "null" string or integer)
    if folder_id is not None:
        if isinstance(folder_id, str) and folder_id.lower() == "null":
            query = query.filter(File.folder_id.is_(None))
        else:
            try:
                folder_id_int = int(folder_id)
                query = query.filter(File.folder_id == folder_id_int)
            except ValueError:
                raise HTTPException(status_code=422, detail="Invalid folder_id parameter")
    else:
        query = query.filter(File.folder_id.is_(None))
    
    files = query.order_by(File.created_at.desc()).offset(skip).limit(limit).all()
    
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
    parent_id: Optional[Union[int, str]] = Query(None),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List folders"""
    query = db.query(Folder).filter(Folder.owner_id == current_user.id)
    
    # Handle parent_id parameter (can be "null" string or integer)
    if parent_id is not None:
        if isinstance(parent_id, str) and parent_id.lower() == "null":
            query = query.filter(Folder.parent_id.is_(None))
        else:
            try:
                parent_id_int = int(parent_id)
                query = query.filter(Folder.parent_id == parent_id_int)
            except ValueError:
                raise HTTPException(status_code=422, detail="Invalid parent_id parameter")
    else:
        query = query.filter(Folder.parent_id.is_(None))
    
    folders = query.order_by(Folder.created_at.desc()).offset(skip).limit(limit).all()
    
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
@app.get("/search", response_model=List[FileResponse])
async def search_files(
    query: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Search files by name"""
    files = db.query(File).filter(
        File.owner_id == current_user.id,
        (File.filename.ilike(f"%{query}%")) | (File.original_filename.ilike(f"%{query}%"))
    ).order_by(File.created_at.desc()).limit(50).all()
    
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
@app.get("/public/files/{file_id}", response_model=FileResponse)
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
    share_request: ShareRequest,
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
    
    db_file.is_public = share_request.is_public
    db.commit()
    db.refresh(db_file)
    
    return {"message": f"File is now {'public' if share_request.is_public else 'private'}", "is_public": share_request.is_public}

# Health check
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Preflight OPTIONS handler
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    """Handle preflight OPTIONS requests"""
    return JSONResponse(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "Authorization, Content-Type, Accept, Origin, X-Requested-With",
            "Access-Control-Allow-Credentials": "true",
        }
    )

# Root endpoint
@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Cloud Drive API",
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
