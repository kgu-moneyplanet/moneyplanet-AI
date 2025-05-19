from datetime import date, datetime
import ulid
import sqlalchemy
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Date, Integer, DateTime
from app.database import Base


class Decision(Base):
    __tablename__ = "Decision"
    id: Mapped[str] = mapped_column(String(length=26), primary_key=True, nullable=False, default=lambda: str(ulid.ULID()))
    user_id: Mapped[str] = mapped_column(String(length=26), nullable=False)
    planet: Mapped[str] = mapped_column(String(length=26), nullable=False)
    gender: Mapped[str] = mapped_column(String(length=26), nullable=False)
    prefer: Mapped[str] = mapped_column(String(length=26), nullable=True)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    job: Mapped[str] = mapped_column(String(length=26), nullable=False)
    tx_date: Mapped[date] = mapped_column(Date, nullable=False)
    amount: Mapped[int] = mapped_column(Integer, nullable=False)
    category_name: Mapped[str] = mapped_column(String(length=26), nullable=False)
    content: Mapped[str] = mapped_column(String(length=255), nullable=True)
    memo: Mapped[str] = mapped_column(String(length=255), nullable=True)
    abc: Mapped[str] = mapped_column(String(length=5), nullable=False)
    reason: Mapped[str] = mapped_column(String(length=255), nullable=True)
    feedback: Mapped[str] = mapped_column(String(length=255), nullable=True)
    create_datetime: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow(),
        server_default=sqlalchemy.text('now()')  # alembic용
    )
    update_datetime: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow(),
        server_default=sqlalchemy.text('now()')  # alembic용
    )
