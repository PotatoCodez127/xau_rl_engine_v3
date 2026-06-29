# whatsapp_manager.py
import time
import requests
from datetime import datetime, timezone, timedelta

# Replace with your Meta/Twilio WhatsApp API credentials
WHATSAPP_API_URL = "https://graph.facebook.com/v17.0/YOUR_PHONE_NUMBER_ID/messages"
HEADERS = {"Authorization": "Bearer YOUR_ACCESS_TOKEN", "Content-Type": "application/json"}

class WhatsAppCopilot:
    def __init__(self):
        # In production, back this with Redis or PostgreSQL
        self.subscribers = {}  # Format: { "phone_number": "last_interaction_utc_timestamp" }
        self.window_limit = timedelta(hours=24)
        self.reminder_threshold = timedelta(hours=23, minutes=30)

    def update_interaction(self, phone_number: str):
        """Called via FastAPI webhook when a user sends a message."""
        self.subscribers[phone_number] = datetime.now(timezone.utc)
        print(f"Window refreshed for {phone_number}. Valid for 24h.")

    def broadcast_signal(self, signal_data: dict):
        """Sends the trading signal to all users within the open window."""
        now = datetime.now(timezone.utc)
        active_users = [
            num for num, last_time in self.subscribers.items()
            if now - last_time < self.window_limit
        ]

        message_body = (
            f"⚜️ XAU_RL_V3 SIGNAL ⚜️\n"
            f"Action: {signal_data['type']}\n"
            f"Entry: {signal_data['entry']:.3f}\n"
            f"SL: {signal_data['sl']:.3f}\n"
            f"TP: {signal_data['tp']:.3f}"
        )

        for user in active_users:
            self._send_message(user, message_body)

    def check_and_send_reminders(self):
        """Runs periodically to warn users their window is closing."""
        now = datetime.now(timezone.utc)
        for num, last_time in self.subscribers.items():
            elapsed = now - last_time
            if self.reminder_threshold <= elapsed < self.window_limit:
                reminder = "⚠️ Your signal window is closing in 30 mins! Reply 'SYNC' to keep it open."
                self._send_message(num, reminder)

    def _send_message(self, to_number: str, text: str):
        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "text",
            "text": {"body": text}
        }
        requests.post(WHATSAPP_API_URL, headers=HEADERS, json=payload)