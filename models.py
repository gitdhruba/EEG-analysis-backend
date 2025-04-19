from sqlalchemy import Integer, Float, String, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass




class Subject(Base):
    __tablename__ = "subjects"

    id : Mapped[int] = mapped_column(Integer, primary_key=True)
    name : Mapped[str] = mapped_column(String, unique=True, nullable=False)
    type : Mapped[int] = mapped_column(Integer, nullable=False)

    eis : Mapped[list["EI"]] = relationship("EI", back_populates="subject", cascade="all")
    psds : Mapped[list["PSD"]] = relationship("PSD", back_populates="subject", cascade="all")
    

    def __init__(self, name : str, type : int):
        self.name = name
        self.type = type

    def __repr__(self) -> str:
        return f"{self.name} -- {self.type}"
    



class EI(Base):
    __tablename__ = "eis"

    id : Mapped[int] = mapped_column(Integer, primary_key=True)
    subject_id : Mapped[int] = mapped_column(ForeignKey("subjects.id"), nullable=False)
    event : Mapped[str] = mapped_column(String, nullable=False)
    value : Mapped[float] = mapped_column(Float)

    subject : Mapped["Subject"] = relationship("Subject", back_populates="eis")


    def __init__(self, sub : int, event : str, ei : float):
        self.subject_id = sub
        self.event = event
        self.value = ei

    def __repr__(self) -> str:
        return f"{self.subject_id} -- {self.event} -- {self.value}"
    



class PSD(Base):
    __tablename__ = "psds"

    id : Mapped[int] = mapped_column(Integer, primary_key=True)
    subject_id : Mapped[int] = mapped_column(ForeignKey("subjects.id"), nullable=False)
    event : Mapped[str] = mapped_column(String, nullable=False)
    band : Mapped[str] = mapped_column(String, nullable=False)
    frequencies : Mapped[list[float]] = mapped_column(ARRAY(Float))
    pxx_values : Mapped[list[float]] = mapped_column(ARRAY(Float))

    subject : Mapped["Subject"] = relationship("Subject", back_populates="psds")


    def __init__(self, sub : int, event : str, band : str, frequencies : list[float], pxx_s : list[float]):
        self.subject_id = sub
        self.event = event
        self.band = band
        self.frequencies = frequencies
        self.pxx_values = pxx_s

    def __repr__(self) -> str:
        return f"{self.subject_id} -- {self.event} -- {self.band} -- {self.frequencies} -- {self.pxx_values}"