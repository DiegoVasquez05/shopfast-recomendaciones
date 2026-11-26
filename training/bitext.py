"""Carga del dataset asignado: Bitext Gen AI Chatbot Customer Support Dataset (Kaggle).

https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset
Archivo: Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
Columnas: flags, instruction, category, intent, response  (~26.900 filas, 27 intents, 11 categorías)

`generate_sample()` produce un CSV sintético con el MISMO esquema para tests/CI
(no reemplaza al dataset real en el informe).
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

REQUIRED = ["instruction", "category", "intent"]


def load_bitext(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"El CSV no tiene las columnas {sorted(missing)} (¿es el dataset Bitext?)")
    df = df.dropna(subset=REQUIRED)
    df["instruction"] = df["instruction"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip().str.lower()
    df["category"] = df["category"].astype(str).str.strip().str.upper()
    return df.drop_duplicates(subset=["instruction", "intent"]).reset_index(drop=True)


def dataset_summary(df: pd.DataFrame) -> dict:
    return {"filas": int(len(df)), "intents": int(df["intent"].nunique()), "categorias": int(df["category"].nunique()),
            "por_categoria": df["category"].value_counts().to_dict(),
            "largo_promedio_palabras": round(float(df["instruction"].str.split().str.len().mean()), 2)}


_EN = {
    "cancel_order": ("ORDER", ["cancel order {n}", "i want to cancel my order", "help me cancel purchase {n}",
                               "how can i cancel the order i made", "i need to cancel an order"]),
    "change_order": ("ORDER", ["change order {n}", "i want to modify my order", "how do i edit my purchase",
                               "update the items of order {n}"]),
    "place_order": ("ORDER", ["i want to buy a phone", "place an order", "how can i purchase a new device",
                              "i'd like to order a router"]),
    "track_order": ("ORDER", ["where is my order {n}", "track order {n}", "check the status of my order",
                              "when will my package arrive"]),
    "create_account": ("ACCOUNT", ["create an account", "i want to open an account", "how do i sign up",
                                   "help me register a new user account"]),
    "delete_account": ("ACCOUNT", ["delete my account", "close my account", "i want to remove my profile",
                                   "how can i terminate my account"]),
    "edit_account": ("ACCOUNT", ["edit my account details", "update my profile", "change my account information",
                                 "modify personal data in my account"]),
    "recover_password": ("ACCOUNT", ["i forgot my password", "reset my password", "recover my pwd",
                                     "can't remember my password"]),
    "registration_problems": ("ACCOUNT", ["i have problems signing up", "registration error",
                                          "i can't register", "issue with the sign-up form"]),
    "switch_account": ("ACCOUNT", ["switch to another account", "change my account to {name}",
                                   "use a different account", "swap accounts"]),
    "check_cancellation_fee": ("CANCEL", ["check cancellation fee", "how much is the early termination fee",
                                          "is there a penalty for canceling", "cancellation charges"]),
    "contact_customer_service": ("CONTACT", ["contact customer service", "what are your support hours",
                                             "how do i reach customer support", "customer service phone number"]),
    "contact_human_agent": ("CONTACT", ["talk to a human", "i want to speak with an agent", "live agent please",
                                        "connect me to a real person"]),
    "delivery_options": ("DELIVERY", ["what delivery options do you have", "shipping methods",
                                      "how can i get my order delivered", "delivery choices"]),
    "delivery_period": ("DELIVERY", ["how long does delivery take", "when will i receive my order",
                                     "delivery time estimate", "how many days for shipping"]),
    "complaint": ("FEEDBACK", ["i want to file a complaint", "i'm very unhappy with the service",
                               "this is unacceptable i want to complain", "lodge a complaint"]),
    "review": ("FEEDBACK", ["i want to leave a review", "submit feedback about your service",
                            "how do i rate the product", "write a review"]),
    "check_invoice": ("INVOICE", ["check my invoice", "how much is my bill", "show invoice {n}",
                                  "what is the amount of my last bill"]),
    "get_invoice": ("INVOICE", ["download my invoice", "send me a copy of the bill", "get invoice {n}",
                                "i need my invoice in pdf"]),
    "newsletter_subscription": ("NEWSLETTER", ["subscribe to the newsletter", "unsubscribe from emails",
                                               "i want to receive your offers", "newsletter sign up"]),
    "check_payment_methods": ("PAYMENT", ["what payment methods do you accept", "can i pay with credit card",
                                          "payment options", "do you accept paypal"]),
    "payment_issue": ("PAYMENT", ["my payment failed", "i was charged twice", "problem with my payment",
                                  "payment was declined"]),
    "check_refund_policy": ("REFUND", ["what is your refund policy", "can i get my money back",
                                       "refund conditions", "how do refunds work"]),
    "get_refund": ("REFUND", ["i want a refund", "request a refund for order {n}", "give me my money back",
                              "how do i get reimbursed"]),
    "track_refund": ("REFUND", ["where is my refund", "track my refund", "status of my reimbursement",
                                "refund not received yet"]),
    "change_shipping_address": ("SHIPPING", ["change my shipping address", "update delivery address",
                                             "i moved, change the address", "modify shipping address"]),
    "set_up_shipping_address": ("SHIPPING", ["set up a shipping address", "add a new delivery address",
                                             "register my address", "create shipping address"]),
}
_PREFIX = ["", "", "hi, ", "hello ", "please ", "can you help me? ", "i need help: ", "hey "]
_SUFFIX = ["", "", "", " please", " asap", "?", " thanks", " right now"]


def generate_sample(n_per_intent: int = 120, seed: int = 0) -> pd.DataFrame:
    """Dataset sintético con el esquema Bitext (solo para pruebas)."""
    rnd = random.Random(seed)
    rows = []
    for intent, (cat, templates) in _EN.items():
        seen = set()
        for _ in range(n_per_intent * 3):
            t = rnd.choice(templates).format(n=rnd.randint(1000, 99999), name=rnd.choice(["ana", "pedro", "luis"]))
            text = (rnd.choice(_PREFIX) + t + rnd.choice(_SUFFIX)).strip()
            if text in seen:
                continue
            seen.add(text)
            rows.append({"flags": "B", "instruction": text, "category": cat, "intent": intent,
                         "response": f"[sample response for {intent}]"})
            if len(seen) >= n_per_intent:
                break
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Genera un CSV sintético con el esquema Bitext (para tests)")
    ap.add_argument("--out", default="data/bitext_sample.csv")
    a = ap.parse_args()
    df = generate_sample()
    df.to_csv(a.out, index=False)
    print(f"{len(df)} filas -> {a.out}")
