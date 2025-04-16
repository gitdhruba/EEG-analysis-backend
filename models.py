from sqlalchemy import Integer, Float, String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Subject(Base):
    __tablename__ = "subjects"

    id : Mapped[int] = mapped_column(Integer, primary_key=True)
    subject_name : Mapped[str] = mapped_column(String, unique=True, nullable=False)
    type : Mapped[int] = mapped_column(Integer, nullable=False)

    def __init__(self, name : str, type : int):
        self.subject_name = name
        self.type = type

    def __repr__(self) -> str:
        return f"{self.subject_name} -- {self.type}"
    

class EI(Base):
    __tablename__ = "eis"

    id : Mapped[int] = mapped_column(Integer, primary_key=True)
    subject_id : Mapped[int] = mapped_column(Integer, nullable=False)
    ei_values : Mapped[list[float]] = mapped_column(ARRAY(Float))

    def __init__(self, sub : int, ei_s : list[float]):
        self.subject_id = sub
        self.ei_values = ei_s

    def __repr__(self) -> str:
        return f"{self.subject_id} -- {self.ei_values}"
    

class PSD(Base):
    __tablename__ = "psds"

    id : Mapped[int] = mapped_column(Integer, primary_key=True)
    subject_id : Mapped[int] = mapped_column(Integer, nullable=False)
    band : Mapped[str] = mapped_column(String, nullable=False)
    pxx_values : Mapped[list[float]] = mapped_column(ARRAY(Float))

    def __init__(self, sub : int, band : str, pxx_s : list[float]):
        self.subject_id = sub
        self.band = band
        self.pxx_values = pxx_s

    def __repr__(self) -> str:
        return f"{self.subject_id} -- {self.band} -- {self.pxx_values}"