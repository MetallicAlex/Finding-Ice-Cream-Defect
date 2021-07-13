import enum
from contextlib import contextmanager
from typing import Union
from datetime import datetime
import json

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext import mutable
from sqlalchemy import null, func, TypeDecorator
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy import Column, Integer, String, DECIMAL, ForeignKey, DateTime, Boolean, Text, Float

engine = sqlalchemy.create_engine('mysql+pymysql://root:admin@localhost/ice_cream')
Base = declarative_base()
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


class IceCream(Base):
    id = Column('ID', Integer, primary_key=True, unique=True, autoincrement=True)
    created_at = Column('CreatedAt', DateTime, nullable=True)
    image = Column('Image', String(128), nullable=True)
    cost = Column('Cost', Float, nullable=True)
    defect = Column('Defect', Boolean, nullable=True)
    runtime = Column('Runtime', Float, nullable=True)
    payload = Column('Payload', Text, nullable=True)


@contextmanager
def get_session():
    session = Session()
    session.expire_on_commit = False
    try:
        yield session
    except:
        session.rollback()
        raise
    else:
        session.commit()
