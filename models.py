from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Boolean, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mysql import BINARY
from datetime import datetime, timezone
import uuid

Base = declarative_base()

# User 테이블
class User(Base):
    __tablename__ = "user"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))  # UUID 사용
    name = Column(String(255), unique=True, nullable=False)
    password = Column(BINARY(60), nullable=False)
    difficulty_lv = Column(Enum('E', 'H'), nullable=True)
    description_st = Column(String(255), nullable=True)
    clr_knowledge = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # User와 UserPhoto 관계 (One-to-Many)
    photos = relationship(
        "UserPhoto",
        back_populates="user",
        cascade="all, delete",
        primaryjoin="User.id == UserPhoto.user_id"
    )

    # User와 GeneratedDescription 관계 (One-to-Many)
    descriptions = relationship(
        "GeneratedDescription",
        back_populates="user",
        cascade="all, delete",
        primaryjoin="User.id == GeneratedDescription.user_id"
    )


# UserPhoto 테이블 (사진 테이블)
class UserPhoto(Base):
    __tablename__ = "userphoto"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # User 관계 설정 (Many-to-One)
    user = relationship(
        "User",
        back_populates="photos",
        primaryjoin="User.id == UserPhoto.user_id"
    )

    # GeneratedDescription 관계 설정 (One-to-Many)
    descriptions = relationship(
        "GeneratedDescription",
        back_populates="photo",
        cascade="all, delete",
        primaryjoin="UserPhoto.id == GeneratedDescription.photo_id"
    )


# Artwork 테이블
class Artwork(Base):
    __tablename__ = "artwork"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    artist = Column(String(255), nullable=False)
    year = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    img_url = Column(String(2083), nullable=True)


# GeneratedDescription 테이블 (AI 생성 설명)
class GeneratedDescription(Base):
    __tablename__ = "generateddescription"

    id = Column(Integer, primary_key=True, autoincrement=True)
    photo_id = Column(Integer, ForeignKey("userphoto.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    artwork_id = Column(Integer, ForeignKey("artwork.id", ondelete="SET NULL"), nullable=True)
    generated_text = Column(Text, nullable=True)
    photo_url = Column(String(2083), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # User 관계 설정 (Many-to-One)
    user = relationship(
        "User",
        back_populates="descriptions",
        primaryjoin="User.id == GeneratedDescription.user_id"
    )

    # UserPhoto 관계 설정 (Many-to-One)
    photo = relationship(
        "UserPhoto",
        primaryjoin="UserPhoto.id == GeneratedDescription.photo_id"
    )

    # Artwork 관계 설정 (Many-to-One)
    artwork = relationship("Artwork")
