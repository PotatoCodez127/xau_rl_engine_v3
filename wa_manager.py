# whatsapp_manager.py
import json
import os
import requests
from datetime import datetime, timezone, timedelta

class WhatsAppCopilot:
    def __init__(self):
        self.api_url = os.getenv("WHATSAPP_API_URL")
        self.token = os.getenv("WHATSAPP_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        
        self.db_file = "subscribers.json"
        self.window_limit = timedelta(hours=24)
        self.reminder_threshold = timedelta(hours=23, minutes=30)
        self.subscribers = self.load_subscribers()

    def load_subscribers(self):
        """Loads JSON and converts stored ISO strings back to datetime objects."""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    data = json.load(f)
                    # Convert strings back to datetime objects
                    return {k: datetime.fromisoformat(v) for k, v in data.items()}
            except Exception as e:
                print(f"[Storage] Error loading JSON: {e}")
        return {}

    def save_subscribers(self):
        """Saves current state, converting datetime objects to ISO strings."""
        with open(self.db_file, 'w') as f:
            # Convert datetime objects to ISO strings for JSON storage
            data_to_save = {k: v.isoformat() for k, v in self.subscribers.items()}
            json.dump(data_to_save, f)

    def update_interaction(self, phone_number: str):
        """Called via FastAPI webhook when a user sends a message."""
        self.subscribers[phone_number] = datetime.now(timezone.utc)
        self.save_subscribers()
        print(f"[WhatsApp] Window refreshed for {phone_number}.")
        self._send_message(phone_number, "System Synced. You are actively receiving XAU Live Signals.")

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
                reminder = "⚠️ Your signal window is closing in less than 30 minutes! Reply 'SYNC' to keep the live feed open."
                self._send_message(num, reminder)
                print(f"[WhatsApp] Reminder dispatched to {num}")

    def _send_message(self, to_number: str, text: str):
        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "text",
            "text": {"body": text}
        }
        try:
            requests.post(self.api_url, headers=self.headers, json=payload)
        except Exception as e:
            print(f"[WhatsApp] Error: {e}")