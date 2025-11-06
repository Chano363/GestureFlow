#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classroom Interaction Controller - Manages classroom interaction functionality
"""

import os
import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict

from utils.config import Config
from utils.enums import InteractionType, GestureType


class InteractionSession:
    """Interaction session class for tracking classroom interactions"""

    def __init__(self, session_id: str, config: Config):
        """Initialize interaction session"""
        self.session_id = session_id
        self.config = config
        self.start_time = datetime.now()
        self.end_time = None
        self.is_active = True

        self.gesture_history: List[Dict] = []
        self.questions: List[Dict] = []
        self.answers: List[Dict] = []
        self.polls: List[Dict] = []
        self.feedback: List[Dict] = []

        self.gesture_counts: Dict[str, int] = defaultdict(int)
        self.interaction_counts: Dict[str, int] = defaultdict(int)
        self.student_participation: Dict[str, Dict] = defaultdict(
            lambda: {
                "gestures": 0,
                "questions": 0,
                "answers": 0,
                "polls": 0,
                "feedback": 0,
            }
        )

        self.event_callbacks: Dict[str, List[Callable]] = defaultdict(list)

        self.lock = threading.Lock()

    def end_session(self):
        with self.lock:
            self.is_active = False
            self.end_time = datetime.now()

    def record_gesture(
        self,
        gesture_type: GestureType,
        confidence: float,
        student_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        if not self.is_active:
            return

        with self.lock:
            gesture_data = {
                "timestamp": datetime.now().isoformat(),
                "gesture_type": gesture_type.name,
                "confidence": confidence,
                "student_id": student_id,
                "metadata": metadata or {},
            }

            self.gesture_history.append(gesture_data)
            self.gesture_counts[gesture_type.name] += 1

            if student_id:
                self.student_participation[student_id]["gestures"] += 1

            self._trigger_event("gesture_recorded", gesture_data)

    def record_question(
        self,
        question: str,
        student_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        if not self.is_active:
            return

        with self.lock:
            question_data = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "student_id": student_id,
                "metadata": metadata or {},
            }

            self.questions.append(question_data)
            self.interaction_counts["questions"] += 1

            if student_id:
                self.student_participation[student_id]["questions"] += 1

            self._trigger_event("question_recorded", question_data)

    def record_answer(
        self,
        answer: str,
        question_id: Optional[str] = None,
        student_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        if not self.is_active:
            return

        with self.lock:
            answer_data = {
                "timestamp": datetime.now().isoformat(),
                "answer": answer,
                "question_id": question_id,
                "student_id": student_id,
                "metadata": metadata or {},
            }

            self.answers.append(answer_data)
            self.interaction_counts["answers"] += 1

            if student_id:
                self.student_participation[student_id]["answers"] += 1

            self._trigger_event("answer_recorded", answer_data)

    def record_poll(
        self,
        poll_question: str,
        options: List[str],
        student_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        if not self.is_active:
            return

        with self.lock:
            poll_data = {
                "timestamp": datetime.now().isoformat(),
                "poll_question": poll_question,
                "options": options,
                "student_id": student_id,
                "metadata": metadata or {},
            }

            self.polls.append(poll_data)
            self.interaction_counts["polls"] += 1

            if student_id:
                self.student_participation[student_id]["polls"] += 1

            self._trigger_event("poll_recorded", poll_data)

    def record_feedback(
        self,
        feedback: str,
        rating: Optional[int] = None,
        student_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        if not self.is_active:
            return

        with self.lock:
            feedback_data = {
                "timestamp": datetime.now().isoformat(),
                "feedback": feedback,
                "rating": rating,
                "student_id": student_id,
                "metadata": metadata or {},
            }

            self.feedback.append(feedback_data)
            self.interaction_counts["feedback"] += 1

            if student_id:
                self.student_participation[student_id]["feedback"] += 1

            self._trigger_event("feedback_recorded", feedback_data)

    def register_event_callback(self, event_type: str, callback: Callable):
        self.event_callbacks[event_type].append(callback)

    def _trigger_event(self, event_type: str, data: Dict):
        for callback in self.event_callbacks[event_type]:
            try:
                callback(data)
            except Exception as e:
                print(f"Error in event callback: {e}")

    def get_session_summary(self) -> Dict:
        with self.lock:
            duration = (self.end_time or datetime.now()) - self.start_time

            return {
                "session_id": self.session_id,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": duration.total_seconds(),
                "is_active": self.is_active,
                "total_gestures": len(self.gesture_history),
                "total_questions": len(self.questions),
                "total_answers": len(self.answers),
                "total_polls": len(self.polls),
                "total_feedback": len(self.feedback),
                "gesture_counts": dict(self.gesture_counts),
                "interaction_counts": dict(self.interaction_counts),
                "student_participation": dict(self.student_participation),
            }

    def to_dict(self) -> Dict:
        with self.lock:
            return {
                "session_id": self.session_id,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "is_active": self.is_active,
                "gesture_history": self.gesture_history,
                "questions": self.questions,
                "answers": self.answers,
                "polls": self.polls,
                "feedback": self.feedback,
                "gesture_counts": dict(self.gesture_counts),
                "interaction_counts": dict(self.interaction_counts),
                "student_participation": dict(self.student_participation),
            }


class InteractionController:
    """Classroom interaction controller for managing classroom interactions"""

    def __init__(self, config: Config):
        self.config = config
        self.current_session: Optional[InteractionSession] = None
        self.session_history: List[InteractionSession] = []

        self.session_timeout = config.get("classroom.session_timeout", 3600)  # 1 hour
        self.max_students = config.get("classroom.max_students", 100)
        self.auto_save_interval = config.get(
            "classroom.auto_save_interval", 300
        )  # 5 minutes
        self.gesture_history_size = config.get("classroom.gesture_history_size", 1000)

        self.data_path = config.get("classroom.data_path", "data/interaction_data.json")
        self.export_format = config.get("classroom.export_format", "json")

        self.auto_save_timer = None
        self._start_auto_save_timer()

    def start_session(self, session_id: Optional[str] = None) -> str:
        if self.current_session and self.current_session.is_active:
            self.end_session()

        if not session_id:
            session_id = f"session_{int(time.time())}"

        self.current_session = InteractionSession(session_id, self.config)

        self._register_default_callbacks()

        return session_id

    def end_session(self):
        if not self.current_session or not self.current_session.is_active:
            return

        self.current_session.end_session()
        self.session_history.append(self.current_session)

        self._save_session_data(self.current_session)

        self.current_session = None

    def record_gesture(
        self,
        gesture_type: GestureType,
        confidence: float,
        student_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        if not self.current_session or not self.current_session.is_active:
            return

        self.current_session.record_gesture(
            gesture_type, confidence, student_id, metadata
        )

    def record_question(
        self,
        question: str,
        student_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        if not self.current_session or not self.current_session.is_active:
            return

        self.current_session.record_question(question, student_id, metadata)

    def record_answer(
        self,
        answer: str,
        question_id: Optional[str] = None,
        student_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        if not self.current_session or not self.current_session.is_active:
            return

        self.current_session.record_answer(answer, question_id, student_id, metadata)

    def record_poll(
        self,
        poll_question: str,
        options: List[str],
        student_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        if not self.current_session or not self.current_session.is_active:
            return

        self.current_session.record_poll(poll_question, options, student_id, metadata)

    def record_feedback(
        self,
        feedback: str,
        rating: Optional[int] = None,
        student_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        if not self.current_session or not self.current_session.is_active:
            return

        self.current_session.record_feedback(feedback, rating, student_id, metadata)

    def get_current_session(self) -> Optional[InteractionSession]:
        return self.current_session

    def get_session_history(self) -> List[InteractionSession]:
        return self.session_history.copy()

    def get_session_summary(self) -> Optional[Dict]:
        if not self.current_session:
            return None

        return self.current_session.get_session_summary()

    def export_data(
        self, session_id: Optional[str] = None, file_path: Optional[str] = None
    ) -> str:
        if session_id:
            session = next(
                (s for s in self.session_history if s.session_id == session_id), None
            )
            if not session:
                raise ValueError(f"Session not found: {session_id}")
        elif self.current_session:
            session = self.current_session
        else:
            raise ValueError("No session to export")

        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"{self.data_path.rsplit('.', 1)[0]}_{session.session_id}_{timestamp}.{self.export_format}"

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if self.export_format == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        elif self.export_format == "csv":
            import pandas as pd

            gestures_df = pd.DataFrame(session.gesture_history)
            gestures_df.to_csv(
                file_path.replace(f".{self.export_format}", "_gestures.csv"),
                index=False,
            )

            questions_df = pd.DataFrame(session.questions)
            questions_df.to_csv(
                file_path.replace(f".{self.export_format}", "_questions.csv"),
                index=False,
            )

            answers_df = pd.DataFrame(session.answers)
            answers_df.to_csv(
                file_path.replace(f".{self.export_format}", "_answers.csv"), index=False
            )

            polls_df = pd.DataFrame(session.polls)
            polls_df.to_csv(
                file_path.replace(f".{self.export_format}", "_polls.csv"), index=False
            )

            feedback_df = pd.DataFrame(session.feedback)
            feedback_df.to_csv(
                file_path.replace(f".{self.export_format}", "_feedback.csv"),
                index=False,
            )

            summary_df = pd.DataFrame([session.get_session_summary()])
            summary_df.to_csv(
                file_path.replace(f".{self.export_format}", "_summary.csv"), index=False
            )
        else:
            raise ValueError(f"Unsupported export format: {self.export_format}")

        return file_path

    def _register_default_callbacks(self):
        self.current_session.register_event_callback(
            "question_recorded", self._schedule_auto_save
        )
        self.current_session.register_event_callback(
            "answer_recorded", self._schedule_auto_save
        )
        self.current_session.register_event_callback(
            "poll_recorded", self._schedule_auto_save
        )
        self.current_session.register_event_callback(
            "feedback_recorded", self._schedule_auto_save
        )

    def _schedule_auto_save(self, data):
        if self.auto_save_timer:
            self.auto_save_timer.cancel()

        self.auto_save_timer = threading.Timer(self.auto_save_interval, self._auto_save)
        self.auto_save_timer.start()

    def _auto_save(self):
        if self.current_session and self.current_session.is_active:
            self._save_session_data(self.current_session)

    def _save_session_data(self, session: InteractionSession):
        try:
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)

            data = {}
            if os.path.exists(self.data_path):
                with open(self.data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            data[session.session_id] = session.to_dict()

            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving session data: {e}")

    def _start_auto_save_timer(self):
        if self.auto_save_interval > 0:
            self.auto_save_timer = threading.Timer(
                self.auto_save_interval, self._auto_save
            )
            self.auto_save_timer.start()

    def __del__(self):
        if self.auto_save_timer:
            self.auto_save_timer.cancel()

        if self.current_session and self.current_session.is_active:
            self.end_session()
