import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

NAMES = (
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "due_date",
    "currency",
    "subtotal",
    "tax",
    "grand_total",
)
MONEY = {"subtotal", "tax", "grand_total", "quantity", "unit_price", "amount"}


class Evidence(BaseModel):
    page_number: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=10000)


class Value(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw: str | None = Field(default=None, max_length=10000)
    normalized: str | None = Field(default=None, max_length=2000)
    status: Literal["present", "missing", "ambiguous"] = "missing"
    evidence: list[Evidence] = Field(default_factory=list, max_length=100)


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: Value = Field(default_factory=Value)
    quantity: Value = Field(default_factory=Value)
    unit_price: Value = Field(default_factory=Value)
    amount: Value = Field(default_factory=Value)


class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fields: dict[str, Value]
    line_items: list[LineItem] = Field(default_factory=list, max_length=1000)
    unsupported_adjustments: bool = False

    @model_validator(mode="after")
    def validate_values(self):
        if set(self.fields) != set(NAMES):
            raise ValueError("Invoice fields must match the invoice schema.")
        entries = list(self.fields.items()) + [
            (k, v) for line in self.line_items for k, v in line
        ]
        for name, value in entries:
            if value.status == "present" and not value.normalized:
                raise ValueError(f"{name} needs a normalized value.")
            if value.normalized and name in MONEY:
                try:
                    d = Decimal(value.normalized)
                    if not d.is_finite() or abs(d) > Decimal("1e15"):
                        raise ValueError("Amount out of range")
                except InvalidOperation:
                    raise ValueError(f"{name} must be a decimal number.")
            if value.normalized and name in {"invoice_date", "due_date"}:
                date.fromisoformat(value.normalized)
            if (
                value.normalized
                and name == "currency"
                and not re.fullmatch("[A-Z]{3}", value.normalized)
            ):
                raise ValueError("Currency must be an explicit three-letter code.")
        return self


class ReviewRequest(BaseModel):
    invoice: Invoice
    action: Literal["save", "approve"] = "save"
    version: int = Field(ge=0)
    acknowledgement: str = Field(default="", max_length=2000)
