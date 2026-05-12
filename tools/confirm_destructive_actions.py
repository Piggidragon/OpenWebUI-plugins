"""
title: Confirm Destructive Action
author: custom
description: >
  Shows the user a native confirmation dialog before a destructive
  action is executed. The model calls this tool and waits for the
  response before continuing.
version: 2.0.0
"""

from pydantic import BaseModel, Field
from typing import Optional

_MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "waiting": "Waiting for confirmation...",
        "title": "Confirmation required",
        "prompt": "Do you really want to continue?",
        "no_dialog": "Dialog not available",
        "confirmed_status": "Confirmed — proceeding.",
        "confirmed_notif": "Action confirmed.",
        "cancelled_status": "Cancelled.",
        "cancelled_notif": "Action cancelled.",
    },
    "de": {
        "waiting": "Warte auf Bestätigung...",
        "title": "Bestätigung erforderlich",
        "prompt": "Möchtest du wirklich fortfahren?",
        "no_dialog": "Dialog nicht verfügbar",
        "confirmed_status": "Bestätigt — fahre fort.",
        "confirmed_notif": "Aktion bestätigt.",
        "cancelled_status": "Abgebrochen.",
        "cancelled_notif": "Aktion abgebrochen.",
    },
    "es": {
        "waiting": "Esperando confirmación...",
        "title": "Confirmación requerida",
        "prompt": "¿Realmente quieres continuar?",
        "no_dialog": "Diálogo no disponible",
        "confirmed_status": "Confirmado — continuando.",
        "confirmed_notif": "Acción confirmada.",
        "cancelled_status": "Cancelado.",
        "cancelled_notif": "Acción cancelada.",
    },
    "fr": {
        "waiting": "En attente de confirmation...",
        "title": "Confirmation requise",
        "prompt": "Voulez-vous vraiment continuer ?",
        "no_dialog": "Dialogue non disponible",
        "confirmed_status": "Confirmé — poursuite.",
        "confirmed_notif": "Action confirmée.",
        "cancelled_status": "Annulé.",
        "cancelled_notif": "Action annulée.",
    },
    "zh": {
        "waiting": "等待确认...",
        "title": "需要确认",
        "prompt": "你确定要继续吗？",
        "no_dialog": "对话框不可用",
        "confirmed_status": "已确认 — 继续执行。",
        "confirmed_notif": "操作已确认。",
        "cancelled_status": "已取消。",
        "cancelled_notif": "操作已取消。",
    },
    "ja": {
        "waiting": "確認待ち...",
        "title": "確認が必要です",
        "prompt": "本当に続行しますか？",
        "no_dialog": "ダイアログが利用できません",
        "confirmed_status": "確認済み — 続行します。",
        "confirmed_notif": "操作を確認しました。",
        "cancelled_status": "キャンセルされました。",
        "cancelled_notif": "操作がキャンセルされました。",
    },
}


class Tools:
    class Valves(BaseModel):
        confirm_button_label: str = Field(
            default="Yes, continue",
            description="Label for the confirm button",
        )
        cancel_button_label: str = Field(
            default="Cancel",
            description="Label for the cancel button",
        )

    def __init__(self):
        self.valves = self.Valves()

    @staticmethod
    def _msg(user_language: Optional[str], key: str) -> str:
        lang = (user_language or "en").lower()
        return _MESSAGES.get(lang, _MESSAGES["en"]).get(
            key, _MESSAGES["en"][key]
        )

    async def confirm_destructive_action(
            self,
            action_description: str,
            consequence: str,
            __event_emitter__=None,
            __event_call__=None,
            __user__: Optional[dict] = None,
    ) -> str:
        """
        Shows the user a confirmation dialog for a destructive action.
        Returns "confirmed" if the user confirms, "cancelled" otherwise.
        Must be called BEFORE executing any action that deletes,
        overwrites, or irreversibly modifies data.

        :param action_description: Brief description of what will be done.
                                   Example: "Delete the file config.yaml"
        :param consequence: What will irreversibly happen if confirmed.
                            Example: "The file cannot be recovered."
        :return: "confirmed" or "cancelled"
        """
        user_language: Optional[str] = None
        if isinstance(__user__, dict):
            user_language = __user__.get("language")

        m = lambda key: self._msg(user_language, key)

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": m("waiting"), "done": False},
                }
            )

        if __event_call__:
            confirmed = await __event_call__(
                {
                    "type": "confirmation",
                    "data": {
                        "title": f"⚠️ {m('title')}",
                        "message": (
                            f"**Action:** {action_description}\n\n"
                            f"**Consequence:** {consequence}\n\n"
                            f"{m('prompt')}"
                        ),
                    },
                }
            )
        else:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": m("no_dialog"), "done": True},
                    }
                )
            return "cancelled"

        if __event_emitter__:
            if confirmed:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": m("confirmed_status"),
                            "done": True,
                        },
                    }
                )
                await __event_emitter__(
                    {
                        "type": "notification",
                        "data": {"type": "success", "content": m("confirmed_notif")},
                    }
                )
            else:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": m("cancelled_status"), "done": True},
                    }
                )
                await __event_emitter__(
                    {
                        "type": "notification",
                        "data": {"type": "warning", "content": m("cancelled_notif")},
                    }
                )

        return "confirmed" if confirmed else "cancelled"
