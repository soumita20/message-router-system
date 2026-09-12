import pandas as pd
import re
import os
import json
from pathlib import Path
from ollama import chat
from functools import lru_cache
from faster_whisper import WhisperModel

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

#Iterating through all the files in the dataset directory
num_files = 0
CONTEXT_FILES = {
    "messages": "messages.csv",
    "sample_messages": "sample_messages.csv",
    "users": "users.csv",
    "groups": "groups.csv",
    "group_members": "group_members.csv",
    "business_accounts": "business_accounts.csv",
    "user_business_history": "user_business_history.csv",
    "message_history": "message_history.csv",
    "message_events": "message_events.csv",
    "images": "images.csv",
    "voice_notes": "voice_notes.csv",
    "daily_notification_summary": "daily_notification_summary.csv",
}

def load_data():
    dataframes = {}
    for key, filename in CONTEXT_FILES.items():
        file_path = DATASET_DIR / filename
        if file_path.exists() and file_path.is_file():
            dataframes[key] = pd.read_csv(file_path)
            print(f"Loaded {filename} with shape: {dataframes[key].shape}")
            print(f"Columns in {filename}: {dataframes[key].columns.tolist()}")
            print(f"First 5 rows of {filename}:")
            print(dataframes[key].head())
        else:
            print(f"Warning: {filename} does not exist in the dataset directory.")
    return dataframes

def validate_dataset():
    if not DATASET_DIR.exists() or not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Error: Dataset directory '{DATASET_DIR}' does not exist or is not a directory.")
    available_files = [file.name for file in DATASET_DIR.iterdir() if file.is_file()]
    missing_files = [file for file in CONTEXT_FILES.values() if file not in available_files]
    if len(missing_files) > 0:
        raise FileNotFoundError(f"The following expected files are missing in the dataset directory: {missing_files}")
    return True

def build_final_dataset(user_message_data):
    messages_df = user_message_data["messages"].copy(deep=True)
    messages_df["forwarded_count"] = pd.to_numeric(messages_df["forwarded_count"])
    messages_df.rename(columns={"created_at":"message_created_at","media_id":"message_media_id","media_type":"message_media_type","forwarded_count":"message_forwarded_count", "conversation_type":"message_conversation_type"}, inplace=True)
    users_df = user_message_data["users"].copy(deep=True)
    users_df.rename(columns={"notifications_dismissed_30d":"total_notifications_dismissed_30d","messages_reported_30d":"total_messages_reported_30d","do_not_disturb_window":"users_do_not_disturb_window", "messages_opened_30d":"users_total_messages_opened_30d","messages_replied_30d":"users_total_messages_replied_30d","notifications_dismissed_30d":"users_total_notifications_dismissed_30d", "messages_reported_30d":"users_total_messages_reported_30d"}, inplace=True)
    messages_df = messages_df.merge(users_df, how="left", on="user_id")
    group_members_df = user_message_data["group_members"].copy(deep=True)
    group_members_df.rename(columns={"role":"group_role","notifications_dismissed_30d":"group_notifications_dismissed_30d","messages_sent_30d":"group_messages_sent_30d", "messages_read_30d":"group_messages_read_30d","replies_sent_30d":"group_replies_sent_30d", "joined_at":"group_joined_at"}, inplace=True)
    groups_df = user_message_data["groups"].copy(deep=True)
    groups_df.rename(columns={"messages_30d":"group_total_messages_30d", "created_at":"group_created_at", "member_count":"group_member_count", "admin_count":"group_admin_count"}, inplace=True)
    group_info_df = group_members_df.merge(groups_df, how="left", on="group_id")
    messages_df = messages_df.merge(group_info_df, how="left", on=["group_id", "user_id"])
    images_df = user_message_data["images"].copy(deep=True)
    images_df.rename(columns={"file_path":"image_file_path"}, inplace=True)
    messages_df = messages_df.merge(images_df, how="left", left_on="message_media_id", right_on="image_id")
    business_acc_df = user_message_data["business_accounts"].copy(deep=True)
    business_acc_df.rename(columns={"display_name":"business_display_name","brand_name":"business_brand_name", "category":"business_category","verified":"is_business_verified","official_domain":"business_official_domain", "domain_used_by_sender":"business_domain_used_by_sender","account_age_days":"business_account_age_days","messages_sent_30d":"business_messages_sent_30d","user_reports_30d":"business_user_reports_30d","domain_used_by_sender_age_days":"business_domain_used_by_sender_age_days"}, inplace=True)
    messages_df = messages_df.merge(business_acc_df, how="left", on="business_id")
    user_business_history_df = user_message_data["user_business_history"].copy(deep=True)
    user_business_history_df.rename(columns={"why_user_knows_account":"why_user_knows_business_account", "last_activity_at":"user_business_last_activity", "allows_promotions":"business_allows_promotions", "promotions_opted_out_at":"user_business_promotions_opted_out_at","activity_count_180d":"user_business_activity_count_180d","messages_opened_30d":"user_business_messages_opened_30d","messages_dismissed_30d":"user_business_messages_dismissed_30d","messages_replied_30d":"user_business_messages_replied_30d","last_reply_at":"user_business_last_reply_at"}, inplace=True)
    messages_df = messages_df.merge(user_business_history_df, how="left", on=["user_id", "business_id"])
    voice_notes_df = user_message_data["voice_notes"].copy(deep=True)
    voice_notes_df.rename(columns={"file_path":"voice_note_file_path"}, inplace=True)
    messages_df = messages_df.merge(voice_notes_df, how="left", left_on="message_media_id", right_on="voice_note_id")
    # Notification load over the latest seven days
    notification_summary = user_message_data["daily_notification_summary"].copy()
    notification_summary["date"] = pd.to_datetime(notification_summary["date"])
    latest_date = notification_summary["date"].max()
    recent_notifications = notification_summary[
        notification_summary["date"] >= latest_date - pd.Timedelta(days=6)
    ]

    notifications_df = (
        recent_notifications.groupby("user_id", as_index=False)
        .agg(
            notifications_sent_7d=("notifications_sent", "sum"),
            notifications_dismissed_7d=("notifications_dismissed", "sum"),
        )
    )
    messages_df = messages_df.merge(notifications_df, how="left", on="user_id")

    messages_df.to_excel(DATASET_DIR / "modified_data" / "final_messages_dataset.xlsx", index=False)

    if(len(messages_df) != len(user_message_data["messages"])):
        raise ValueError(
            "A merge changed the messages dataset. Check for duplicate context records."
        )
    return messages_df

def build_historical_dataset(user_message_data):
    message_history_df = user_message_data["message_history"].copy(deep=True)
    message_history_df["created_at"] = pd.to_datetime(message_history_df["created_at"])
    message_history_df.rename(columns={"created_at":"message_history_created_at","media_id":"message_history_media_id","media_type":"message_history_media_type","forwarded_count":"message_history_forwarded_count", "conversation_type":"message_history_conversation_type"}, inplace=True)
    message_history_df = message_history_df.merge(user_message_data["images"], how="left", left_on="message_history_media_id", right_on="image_id")
    message_history_df.rename(columns={"file_path":"image_file_path"}, inplace=True)
    message_history_df = message_history_df.merge(user_message_data["voice_notes"], how="left", left_on="message_history_media_id", right_on="voice_note_id")
    message_history_df.rename(columns={"file_path":"voice_note_file_path"}, inplace=True)
    message_events_df = user_message_data["message_events"].copy(deep=True)
    message_events_df.rename(columns={"reaction_time_minutes":"message_event_reaction_time_minutes","notification_dismissed":"message_event_notification_dismissed","muted_after_message":"message_event_muted_after_message","message_reported":"message_event_reported"}, inplace=True)
    message_history_df = message_history_df.merge(message_events_df, how="left", on=["message_id","user_id"])
    cols_to_fill = ["message_opened","message_replied","message_event_notification_dismissed", "message_event_muted_after_message", "message_event_reported"]
    message_history_df[cols_to_fill] = message_history_df[cols_to_fill].fillna(0).astype(int)

    message_history_df.to_excel(DATASET_DIR / "modified_data" /"historical_messages_dataset.xlsx", index=False)

    if(len(message_history_df) != len(user_message_data["message_history"])):
            raise ValueError(
                "A merge changed the message history dataset. Check for duplicate context records."
            )

    return message_history_df

def build_interaction_summary(historic_messages_df,group_by_cols, type_of_interaction:str, message_summary_df):
    original_length = len(message_summary_df)
    interaction_summary = historic_messages_df.groupby(group_by_cols).agg(
        sent=("message_id", "count"),
        opened=("message_opened", "sum"),
        replied=("message_replied", "sum"),
        dismissed=("message_event_notification_dismissed", "sum"),
        muted=("message_event_muted_after_message", "sum"),
        reported=("message_event_reported", "sum")).reset_index()

    interaction_summary.rename(columns = {
        "sent": f"total_{type_of_interaction}_messages_sent",
        "opened": f"total_{type_of_interaction}_messages_opened",
        "replied": f"total_{type_of_interaction}_messages_replied",
        "dismissed": f"total_{type_of_interaction}_messages_dismissed",
        "muted": f"total_{type_of_interaction}_messages_muted",
        "reported": f"total_{type_of_interaction}_messages_reported"
    }, inplace=True)

    message_summary_df = message_summary_df.merge(interaction_summary, how="left", on=group_by_cols)

    if len(message_summary_df) != original_length:
        raise ValueError(
            f"A merge changed the message summary dataset when adding {type_of_interaction} interaction summary. Check for duplicate context records."
        )

    return message_summary_df

VISION_MODEL = "qwen2.5vl:7b"
_whisper_model = None

def get_whisper_model():
    global _whisper_model

    if _whisper_model is None:
        _whisper_model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
        )

    return _whisper_model

@lru_cache(maxsize=None)
def extract_image_content(relative_path: str) -> str:
    image_path = DATASET_DIR / relative_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    prompt = """
            Read this WhatsApp image/poster.
            Extract important text, dates, deadlines, amounts, links, and requested actions.
            Classify it as urgent, event, payment, promotion, scam, or normal update.
            Return plain text only.

            """.strip()

    response = chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }
        ],
        options={"temperature": 0, "num_ctx":8192},
    )

    return response.message.content.strip()

@lru_cache(maxsize=None)
def transcribe_voice_note(relative_path):
    voice_path = DATASET_DIR / relative_path

    if not voice_path.exists():
        raise FileNotFoundError(f"Voice note not found: {voice_path}")

    whisper_model = get_whisper_model()

    segments, _ = whisper_model.transcribe(
        str(voice_path),
        beam_size=5,
    )

    return " ".join(segment.text.strip() for segment in segments).strip()

def extract_media_content(message_row):
    media_type = message_row.get("message_media_type")

    if pd.isna(media_type):
        return {
            "media_extracted_text": "",
            "media_extraction_status": "not_needed",
        }

    media_type = str(media_type).lower()

    try:
        if media_type == "image":
            relative_path = message_row.get("image_file_path")

            if pd.isna(relative_path):
                return {
                    "media_extracted_text": "",
                    "media_extraction_status": "failed_missing_image_path",
                }

            return {
                "media_extracted_text": extract_image_content(str(relative_path)),
                "media_extraction_status": "image_success",
            }

        if media_type in {"voice", "voice_note"}:
            relative_path = message_row.get("voice_note_file_path")

            if pd.isna(relative_path):
                return {
                    "media_extracted_text": "",
                    "media_extraction_status": "failed_missing_voice_path",
                }

            return {
                "media_extracted_text": transcribe_voice_note(str(relative_path)),
                "media_extraction_status": "voice_success",
            }

        return {
            "media_extracted_text": "",
            "media_extraction_status": "not_needed",
        }

    except Exception as error:
        return {
            "media_extracted_text": "",
            "media_extraction_status": f"failed: {error}",
        }

def convert_message_to_signals(message_row):
    raw_text = message_row.get("combined_text", "")
    text = "" if pd.isna(raw_text) else str(raw_text).lower()
    user_id = str(message_row.get("user_id") or "").lower()
    urgent_texts = [r"\burgent\b", r"\bimmediately\b", r"\btoday\b", r"\bdeadline\b", r"\bemergency\b", r"\bexpire[sd]?\b", r"\blast chance\b"]
    payment_texts = [r"\bpay\b", r"\bpayment\b", r"\bdue\b", r"\binvoice\b", r"\bbill\b", r"\bfee\b", r"\brefund\b", r"\bamount\b"]
    event_texts = [r"\bmeeting\b", r"\bevent\b", r"\bclass\b", r"\bschool\b", r"\bschedule\b", r"\bappointment\b", r"\btomorrow\b", r"\bvenue\b"]
    promotion_texts = [r"\bsale\b",r"\boffer\b", r"\bdiscount\b",r"\bcoupon\b",r"\bdeal\b",r"\blimited.time\b",r"\bfree delivery\b"]
    greeting_texts = [r"\bgood morning\b",r"\bhappy birthday\b",r"\bcongratulations\b",r"\bhappy \w+"]
    forward_texts = [r"\bforward\b", r"\bshare\b",r"\bsend this\b",r"\bpass this on\b"]
    scam_texts = [r"\botp\b",r"\bpin\b",r"\bpassword\b",r"\bverify your account\b",r"\baccount (will be )?blocked\b",r"\bclick (the )?link\b",r"\bpay small (fee|amount)\b", r"\bclaim your (prize|reward)\b", r"\bselected for\b.*\bprize\b"]

    def contains_any(texts:list[str]):
         return any(re.search(row_text,text) is not None for row_text in texts)

    return {
        "has_direct_mention": f"@{user_id}" in text,
        "has_urgent_language": contains_any(urgent_texts),
        "has_payment_language": contains_any(payment_texts),
        "has_event_language": contains_any(event_texts),
        "has_promotion_language": contains_any(promotion_texts),
        "has_greeting_language": contains_any(greeting_texts),
        "has_forward_language": contains_any(forward_texts),
        "has_scam_language": contains_any(scam_texts),
        "is_highly_forwarded": message_row["message_forwarded_count"] >= 3,
    }

class NotificationRouterAgent:
    VALID_ACTIONS = {"notify", "digest", "mute"}

    VALID_MESSAGE_TYPES = {
        "personal",
        "urgent",
        "event",
        "payment",
        "business_update",
        "promotion",
        "greeting",
        "forward",
        "spam",
        "scam",
        "unknown"
    }

    def __init__(self, historic_messages_df):
        self.historic_messages_df = historic_messages_df.copy()
        self.model_name = "qwen2.5:14b"

    @staticmethod
    def is_present(value):
        return pd.notna(value) and str(value).strip() != ""

    @staticmethod
    def get_number(message_row, column_name):
        value = message_row.get(column_name, 0)

        if pd.isna(value):
            return 0.0

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def get_flag(message_row, column_name):
        value = message_row.get(column_name, False)
        return False if pd.isna(value) else bool(value)

    def retrieve_history(self, message_row):
        """Agent memory tool: get history from the closest matching source."""
        history_df = self.historic_messages_df[self.historic_messages_df["user_id"] == message_row["user_id"]].copy()

        if self.is_present(message_row.get("business_id")):
            history_df = history_df[history_df["business_id"] == message_row["business_id"]]

        elif self.is_present(message_row.get("group_id")):
            history_df = history_df[history_df["group_id"] == message_row["group_id"]]

        elif self.is_present(message_row.get("sender_user_id")):
            history_df = history_df[history_df["sender_user_id"] == message_row["sender_user_id"]]

        return history_df

    def classify_type_from_signals(self, message_row):
        conversation_type = str(message_row.get("message_conversation_type", "")).lower()

        if self.get_flag(message_row, "has_scam_language"):
            return "scam"

        if (self.get_flag(message_row, "has_forward_language") and self.get_flag(message_row, "is_highly_forwarded")):
            return "spam"

        if self.get_flag(message_row, "has_promotion_language"):
            return "promotion"

        if self.get_flag(message_row, "has_payment_language"):
            return "payment"

        if self.get_flag(message_row, "has_event_language"):
            return "event"

        if self.get_flag(message_row, "has_greeting_language"):
            return "greeting"

        if self.get_flag(message_row, "has_forward_language"):
            return "forward"

        if self.get_flag(message_row, "has_urgent_language"):
            return "urgent"

        if conversation_type == "business":
            return "business_update"

        if conversation_type == "personal":
            return "personal"

        return "unknown"

    def guardrail_decision(self, message_row):
        """Agent safety and user-preference tool."""
        text = str(message_row.get("combined_text") or "").lower()

        has_scam = self.get_flag(message_row, "has_scam_language")
        has_promotion = self.get_flag(message_row,"has_promotion_language")
        has_direct_mention = self.get_flag(message_row,"has_direct_mention")
        has_urgent = self.get_flag(message_row, "has_urgent_language")

        is_forward_chain = (self.get_flag(message_row, "has_forward_language") and self.get_flag(message_row, "is_highly_forwarded"))

        group_muted = self.get_number(message_row, "group_muted_by_user") == 1

        opted_out = self.is_present(message_row.get("user_business_promotions_opted_out_at"))

        promotion_not_allowed = (str(message_row.get("message_conversation_type", "")).lower() == "business" and self.get_number(message_row, "business_allows_promotions") == 0)

        historical_reports = (self.get_number(message_row, "total_user_messages_reported") + self.get_number(message_row, "total_group_messages_reported") + self.get_number(message_row, "total_business_messages_reported"))

        suspicious_request = any(phrase in text for phrase in ["otp","password","pin","click link","verify your account","account blocked","pay small fee"])

        if has_scam and (suspicious_request or historical_reports > 0):
            return {"action": "mute", "message_type": "scam", "reason": "The message contains scam-like language or a previously reported risk pattern.", "confidence": 0.96}

        if is_forward_chain:
            return {"action": "mute", "message_type": "spam", "reason": "This is a heavily forwarded chain-style message.", "confidence": 0.91}

        if has_promotion and (opted_out or promotion_not_allowed):
            return {"action": "mute", "message_type": "promotion", "reason": "The recipient has opted out of or does not allow this promotion.", "confidence": 0.94}

        if group_muted and not (has_direct_mention and has_urgent):
            return {"action": "mute", "message_type": self.classify_type_from_signals(message_row), "reason": "The message is from a group muted by the recipient.", "confidence": 0.88}

        return None

    def should_ask_model(self, message_row):
        """Avoid slow LLM calls for clear routine messages."""
        media_type = str(message_row.get("message_media_type") or "").lower()

        has_any_signal = any(
            self.get_flag(message_row, column)
            for column in ["has_urgent_language", "has_payment_language", "has_event_language", "has_promotion_language", "has_greeting_language", "has_forward_language", "has_scam_language", "has_direct_mention"])

        return media_type in {"image", "voice", "voice_note"} or not has_any_signal

    def ask_ollama(self, message_row, history_df):
        """Agent reasoning tool."""
        recent_history = history_df.sort_values("message_history_created_at", ascending=False).head(3)

        history_examples = []

        for _, history_row in recent_history.iterrows():
            history_examples.append(
                {
                    "message_id": history_row["message_id"],
                    "text": str(history_row.get("message_text") or "")[:250],
                    "opened": int(history_row["message_opened"]),
                    "replied": int(history_row["message_replied"]),
                    "dismissed": int(history_row["message_event_notification_dismissed"]),
                    "muted": int(history_row["message_event_muted_after_message"]),
                    "reported": int(history_row["message_event_reported"]),
                }
            )

        context = {
            "conversation_type": message_row.get("message_conversation_type"),
            "group_type": message_row.get("group_type"),
            "group_muted_by_user": message_row.get("group_muted_by_user"),
            "business_name": message_row.get("business_display_name"),
            "business_verified": message_row.get("is_business_verified"),
            "known_business_relationship": message_row.get("why_user_knows_business_account"),
            "allows_promotions": message_row.get("business_allows_promotions"),
            "forwarded_count": message_row.get("message_forwarded_count"),
            "historical_examples": history_examples
        }

        prompt = f"""
                    You are a WhatsApp notification-routing agent.

                    Choose one action: notify, digest, or mute.
                    Choose one message type: personal, urgent, event, payment,
                    business_update, promotion, greeting, forward, spam, scam, unknown.

                    Rules:
                    - clear scams and unsafe content must be muted.
                    - safe but non-urgent content should be digested.
                    - notify only when interruption is justified.
                    - use the supplied user history for personalization.

                    Message:
                    {str(message_row.get("combined_text") or "")[:1200]}

                    Context:
                    {json.dumps(context, default=str)}

                    Return only JSON:
                    {{
                    "action": "...",
                    "message_type": "...",
                    "reason": "short sentence",
                    "confidence": 0.0
                    }}
                    """.strip()

        response = chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={
                "temperature": 0,
                "num_ctx": 8192,
            },
        )

        result = json.loads(response.message.content)

        action = result.get("action", "digest")
        message_type = result.get("message_type", "unknown")

        if action not in self.VALID_ACTIONS:
            action = "digest"

        if message_type not in self.VALID_MESSAGE_TYPES:
            message_type = "unknown"

        try:
            confidence = float(result.get("confidence", 0.60))
        except (TypeError, ValueError):
            confidence = 0.60

        return {
            "action": action,
            "message_type": message_type,
            "reason": str(
                result.get(
                    "reason",
                    "The agent assessed the message using available context.",
                )
            ),
            "confidence": max(0.0, min(confidence, 1.0)),
        }

    def fallback_decision(self, message_row):
        """Fast decision for clear non-risk messages."""
        conversation_type = str(message_row.get("message_conversation_type", "")).lower()

        message_type = self.classify_type_from_signals(message_row)

        if (self.get_flag(message_row, "has_direct_mention") and self.get_flag(message_row, "has_urgent_language")):
            return { "action": "notify", "message_type": "urgent", "reason": "The recipient was directly mentioned in a time-sensitive message.", "confidence": 0.89 }

        if (conversation_type == "business" and self.get_number(message_row, "is_business_verified") == 1 and self.is_present(message_row.get("why_user_knows_business_account")) and message_type in {"payment", "event", "business_update"}):
            return {"action": "notify", "message_type": message_type, "reason": "A verified business with an existing user relationship sent an important update.", "confidence": 0.82}

        if conversation_type == "group" and message_type in {"urgent", "event", "payment"}:
            return {"action": "notify", "message_type": message_type, "reason": "This is a relevant time-sensitive group update.", "confidence": 0.76}

        if conversation_type == "personal" and message_type == "personal":
            return {"action": "notify", "message_type": "personal", "reason": "This is a safe personal message.", "confidence": 0.67}

        return {"action": "digest", "message_type": message_type, "reason": "The message appears safe but does not require an immediate interruption.", "confidence": 0.68}

    def select_evidence(self, history_df, action):
        """Agent evidence tool."""
        if history_df.empty:
            return "none"

        evidence_df = history_df.copy()

        if action == "mute":
            evidence_df["evidence_score"] = (evidence_df["message_event_reported"] * 5 + evidence_df["message_event_muted_after_message"] * 3 + evidence_df["message_event_notification_dismissed"] * 2)
        else:
            evidence_df["evidence_score"] = (evidence_df["message_replied"] * 3 + evidence_df["message_opened"] * 2 - evidence_df["message_event_notification_dismissed"])

        evidence_df = evidence_df.sort_values(["evidence_score", "message_history_created_at"],ascending=[False, False])

        evidence_ids = evidence_df.head(2)["message_id"].tolist()

        return ";".join(evidence_ids) if evidence_ids else "none"

    def route(self, message_row):
        history_df = self.retrieve_history(message_row)

        decision = self.guardrail_decision(message_row)

        if decision is None:
            if self.should_ask_model(message_row):
                try:
                    decision = self.ask_ollama(message_row, history_df)
                except Exception:
                    decision = self.fallback_decision(message_row)
            else:
                decision = self.fallback_decision(message_row)

        decision["evidence_message_ids"] = self.select_evidence(
            history_df,
            decision["action"],
        )

        return decision

#Load messages.csv file
def main()-> None:
        #Validate dataset
        validate_dataset()

        #Load all CSV files into a dictionary of DataFrames
        user_message_data = load_data()

        #Load sample_messages.csv file
        print("Sample messages data:")
        print(user_message_data["sample_messages"].head())

        #Combine all the data into one dataset
        message_summary_df = build_final_dataset(user_message_data)
        print("\nMessage summary data:")
        print(message_summary_df.head())
        print("\nColumns in message summary data:")
        print(message_summary_df.columns.tolist())
        print("\nShape of message summary data:", message_summary_df.shape)

        #Build historical dataset
        historic_messages_df = build_historical_dataset(user_message_data)
        print("\nHistorical messages data:")
        print(historic_messages_df.head())
        print("\nColumns in historical messages data:")
        print(historic_messages_df.columns.tolist())
        print("\nShape of historical messages data:", historic_messages_df.shape)

        #Merge historical interaction summary with final messages dataset
        #1. Group interaction summary
        message_summary_df = build_interaction_summary(historic_messages_df, ["group_id", "user_id"], "group", message_summary_df)
        #2. Historical user interaction summary
        message_summary_df = build_interaction_summary(historic_messages_df, ["sender_user_id","user_id"], "user", message_summary_df)
        #3. Business interaction summary
        message_summary_df = build_interaction_summary(historic_messages_df, ["business_id", "user_id"], "business", message_summary_df)

        #Save the final message summary dataset to an Excel file
        message_summary_df.to_excel(DATASET_DIR / "modified_data" / "final_dataset_with_interaction_summary.xlsx", index=False)

        #Lets see how the final message summary dataset looks like
        print("\nFinal message summary data with interaction summary:")
        print(message_summary_df.head())
        print("\nColumns in final message summary data with interaction summary:")
        print(message_summary_df.columns.tolist())
        print("\nShape of final message summary data with interaction summary:", message_summary_df.shape)

        #Extracting information from media files
        media_results_df = message_summary_df.apply(extract_media_content, axis=1, result_type="expand")
        message_summary_df = pd.concat([message_summary_df, media_results_df], axis=1)
        message_summary_df["combined_text"] = message_summary_df["message_text"].fillna("").astype(str).str.strip() + "\n" + message_summary_df["media_extracted_text"].fillna("").astype(str).str.strip()
        text_signals_df = message_summary_df.apply(convert_message_to_signals, axis=1, result_type="expand")
        message_summary_df = pd.concat([message_summary_df, text_signals_df], axis=1)
        print("\nMedia extraction status:")
        print(message_summary_df["media_extraction_status"].value_counts())

        #Save the final message summary dataset to an Excel file
        message_summary_df.to_excel(DATASET_DIR / "modified_data" / "final_dataset_with_interaction_summary.xlsx", index=False)

        #Lets see how the final message summary dataset looks like after adding text classifications
        print("\nFinal message summary data with interaction summary:")
        print(message_summary_df.head())
        print("\nColumns in final message summary data with interaction summary:")
        print(message_summary_df.columns.tolist())
        print("\nShape of final message summary data with interaction summary:", message_summary_df.shape)

        #Agent
        router_agent = NotificationRouterAgent(historic_messages_df)
        agent_results_df = message_summary_df.apply(router_agent.route, axis=1, result_type="expand")
        output_df = pd.concat([message_summary_df[["message_id"]].reset_index(drop=True), agent_results_df.reset_index(drop=True)], axis=1)
        output_df = output_df[["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]]

        if len(output_df)!=len(message_summary_df):
            raise ValueError(f"output.csv must contain exactly {len(message_summary_df)} messages.")

        output_df.to_csv(DATASET_DIR / "output.csv", index=False)

        #Predictions
        print(output_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
    