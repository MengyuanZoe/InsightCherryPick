import datetime
from sqlalchemy import (create_engine, Column, Integer, String, Float, Text,
                        BigInteger, DateTime, ForeignKey, Boolean)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import config


# Database Configureation Constants
HOST = config.DATABASE_HOST
USERNAME = config.DATABASE_USERNAME
PASSWORD = config.DATABASE_PASSWORD
DBNAME = config.DATABASE_NAME
PORT = config.DATABASE_PORT

# Configuring DB Connection
connection_sting = "postgresql://{}:{}@{}:{}/{}".format(USERNAME,PASSWORD,HOST,PORT,DBNAME)
engine = create_engine(connection_sting)
Base = declarative_base()


class InstagramUser(Base):
            
    __tablename__ = 'instagram_user'
    
    id = Column(Integer,primary_key=True)
    instagram_id = Column(BigInteger,unique=True,nullable=False)
    instagram_username = Column(String(31),unique=True,nullable=False)
    email_address = Column(String(255))
    bio = Column(Text)
    num_followers = Column(Integer)
    num_following = Column(Integer)
    num_posts = Column(Integer)
    last_updated = Column(DateTime)
    profile_img_url = Column(String(2084))

    # Relationships
    comment = relationship("Comment", uselist=False, back_populates="instagram_user")
    media = relationship("Media", uselist=False, back_populates="instagram_user")
    product = relationship("Product", uselist=False, back_populates="instagram_user")

class Media(Base):

    __tablename__ = 'media'

    id = Column(Integer,primary_key=True)
    media_id = Column(BigInteger,unique=True,nullable=False)
    instagram_id = Column(BigInteger, ForeignKey('instagram_user.instagram_id'),nullable=False)
    created_time = Column(Integer, nullable=False)
    image_url = Column(Text)
    caption = Column(Text)
    num_likes = Column(Integer)
    num_comments = Column(Integer)
    location = Column(Text)
    product_id = Column(Integer, ForeignKey('product.id'),nullable=True)
    product_tagged = Column(Boolean,default=False)

    # Relationships
    instagram_user = relationship("InstagramUser", back_populates="media")
    comment = relationship("Comment", uselist=False, back_populates="media")

class Comment(Base):
    
    __tablename__ = 'comment'
    
    id = Column(Integer,primary_key=True)
    comment_id  = Column(BigInteger,unique=True,nullable=False)
    instagram_id = Column(BigInteger, ForeignKey('instagram_user.instagram_id'), nullable=False)
    instagram_username = Column(String(31),nullable=False)
    media_id = Column(BigInteger,ForeignKey('media.media_id'),nullable=False)
    created_at = Column(Integer, nullable=False)
    comment_text = Column(Text)
    purchase_intent_score = Column(Float)

    # Relationships
    instagram_user = relationship("InstagramUser", back_populates="comment")
    media = relationship("Media", back_populates="comment")

class TargetAccount(Base):

    __tablename__ = "target_account"

    id = Column(Integer,primary_key=True)
    instagram_id = Column(BigInteger, ForeignKey('instagram_user.instagram_id'),unique=True, nullable=False)
    category = Column(String(40),nullable=False)

class Product(Base):
    __tablename__ = "product"

    id = Column(Integer,primary_key=True)
    instagram_id = Column(BigInteger,ForeignKey('instagram_user.instagram_id'),nullable=False)
    product_name = Column(Text)
    category = Column(Text)

    instagram_user = relationship("InstagramUser", back_populates="product")


Base.metadata.create_all(engine)
