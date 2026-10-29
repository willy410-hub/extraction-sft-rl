"""The extraction contract: which fields the model must return, and their types.

Every other module (generator, verifier, prompts) reads from this file, so the
schema can never silently drift between data generation and grading.
"""

FIELDS = ["invoice_id", "date", "amount", "currency", "customer_id", "customer_name"]

# Fields that may legitimately be absent from the source text. When absent,
# the correct answer is JSON null -- a model that invents a value is wrong.
OPTIONAL_FIELDS = ["customer_id", "customer_name"]

CURRENCIES = ["USD", "EUR", "EGP", "GBP"]

FIELD_TYPES = {
    "invoice_id": str,
    "date": str,  # ISO 8601, YYYY-MM-DD
    "amount": float,
    "currency": str,  # ISO 4217 code
    "customer_id": str,
    "customer_name": str,
}
