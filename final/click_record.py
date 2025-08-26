# -*- coding: utf-8 -*-
"""
基于Kivy的鼠标事件记录程序，适用于安卓APK打包。
- 记录鼠标左/中/右键按下与释放事件
- 记录UTC纳秒级时间戳（numpy.datetime64）
- 双击右键作为实验段落的开始/结束标记
- 标记前后各保留2秒数据
"""
import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.clock import Clock
import numpy as np
import datetime
import json
import rerun as rr

# Configuration
PRE_POST_SECONDS = 2  # Seconds to preserve before and after markers
DOUBLE_CLICK_INTERVAL = 0.4  # Maximum double-click interval (seconds)

# Button mapping
BUTTON_MAP = { 'left': 'left', 'right': 'right', 'middle': 'middle' }

class MouseRecorder(App):
    def build(self):
        self.label = Label(text="Waiting for double right-click to start experiment...", font_size=32)
        Window.bind(on_mouse_down=self.on_mouse_down, on_mouse_up=self.on_mouse_up)
        self.events = []  # All events
        self.segments = []  # Experiment segments (start, end indices)
        self.last_right_click_time = None
        self.in_segment = False
        self.last_button_pressed = None  # Track last button pressed
        
        # Initialize rerun logging
        self.init_rerun_logging()
        
        return self.label

    def init_rerun_logging(self):
        """Initialize rerun logging structure"""
        rr.log("experiment/info", rr.TextLog("Mouse Click Recorder - PC Version"))
        rr.log("experiment/config", rr.TextLog(f"Pre/Post seconds: {PRE_POST_SECONDS}, Double-click interval: {DOUBLE_CLICK_INTERVAL}"))
        rr.log("experiment/state", rr.TextLog("waiting"))

    def get_utc_ns(self):
        now = datetime.datetime.utcnow()
        ns = int(now.timestamp() * 1e9)
        return np.datetime64(ns, 'ns')

    def on_mouse_down(self, window, x, y, button, modifiers):
        self.record_event('down', button)
        self.check_double_right_click(button)
        self.update_button_display(button)

    def on_mouse_up(self, window, x, y, button, modifiers):
        self.record_event('up', button)

    def record_event(self, action, button):
        btn = BUTTON_MAP.get(button, button)
        ts = self.get_utc_ns()
        event_data = {
            'action': action,
            'button': btn,
            'timestamp': str(ts)
        }
        self.events.append(event_data)
        
        # Log to rerun
        self.log_to_rerun(action, btn, ts)

    def log_to_rerun(self, action, button, timestamp):
        """Log mouse events to rerun for visualization"""
        # Convert numpy datetime64 to seconds for rerun timeline
        # time_sec = float(timestamp.astype('datetime64[ns]').astype(np.int64)) / 1e9
        
        # Log the event as a scalar with timeline
        rr.set_time("timeline", timestamp=timestamp)
        
        # Log button action as text
        rr.log(f"mouse_events/{button}/{action}", rr.TextLog(f"{button} {action}"))
        
        # Log as a scalar value for timeline visualization
        # action_value = 1.0 if action == 'down' else 0.0
        # rr.log(f"mouse_timeline/{button}", rr.Scalars(action_value))
        
        # Log experiment state
        if hasattr(self, 'in_segment'):
            rr.log("experiment/state", rr.TextLog("recording" if self.in_segment else "waiting"))

    def update_button_display(self, button):
        """Update display to show the last button pressed"""
        btn = BUTTON_MAP.get(button, button)
        self.last_button_pressed = btn
        if self.in_segment:
            self.label.text = f"Experiment in progress... (Last click: {btn})\n(Double right-click to end)"
        else:
            self.label.text = f"Waiting for double right-click to start experiment...\n(Last click: {btn})"

    def check_double_right_click(self, button):
        if button != 'right':
            return
        now = datetime.datetime.utcnow().timestamp()
        if self.last_right_click_time and (now - self.last_right_click_time) < DOUBLE_CLICK_INTERVAL:
            # Detected double right-click
            if not self.in_segment:
                self.start_segment()
            else:
                self.end_segment()
            self.last_right_click_time = None
        else:
            self.last_right_click_time = now

    def start_segment(self):
        self.in_segment = True
        self.segment_start_idx = len(self.events) - 1
        if self.last_button_pressed:
            self.label.text = f"Experiment in progress... (Last click: {self.last_button_pressed})\n(Double right-click to end)"
        else:
            self.label.text = "Experiment in progress... (Double right-click again to end)"
        
        # Log experiment start to rerun
        current_time = float(self.get_utc_ns().astype('datetime64[ns]').astype(np.int64)) / 1e9
        rr.set_time("timeline", timestamp=current_time)
        rr.log("experiment/events", rr.TextLog("EXPERIMENT_START"))
        rr.log("experiment/state", rr.TextLog("recording"))

    def end_segment(self):
        self.in_segment = False
        segment_end_idx = len(self.events) - 1
        self.segments.append((self.segment_start_idx, segment_end_idx))
        self.label.text = "Experiment ended, data saved."
        
        # Log experiment end to rerun
        current_time = float(self.get_utc_ns().astype('datetime64[ns]').astype(np.int64)) / 1e9
        rr.set_time("timeline", timestamp=current_time)
        rr.log("experiment/events", rr.TextLog("EXPERIMENT_END"))
        rr.log("experiment/state", rr.TextLog("waiting"))
        
        Clock.schedule_once(lambda dt: self.save_and_exit(), 1.5)

    def save_and_exit(self):
        # Extract all segments and 2 seconds of data before and after
        all_indices = set()
        for start, end in self.segments:
            t0 = np.datetime64(self.events[start]['timestamp'])
            t1 = np.datetime64(self.events[end]['timestamp'])
            # 2 seconds before and after
            pre = t0 - np.timedelta64(PRE_POST_SECONDS, 's')
            post = t1 + np.timedelta64(PRE_POST_SECONDS, 's')
            for i, e in enumerate(self.events):
                t = np.datetime64(e['timestamp'])
                if pre <= t <= post:
                    all_indices.add(i)
        selected = [self.events[i] for i in sorted(all_indices)]
        
        # Save to JSON file
        # with open('mouse_record.json', 'w', encoding='utf-8') as f:
        #     json.dump(selected, f, ensure_ascii=False, indent=2)
        
        # Log final summary to rerun
        current_time = float(self.get_utc_ns().astype('datetime64[ns]').astype(np.int64)) / 1e9
        rr.set_time("timeline", timestamp=current_time)
        rr.log("experiment/summary", rr.TextLog(f"Total events: {len(self.events)}, Selected events: {len(selected)}, Segments: {len(self.segments)}"))
        rr.log("experiment/events", rr.TextLog("DATA_SAVED"))
        
        self.stop()

if __name__ == '__main__':
    rr.init("click")
    rr.spawn()
    MouseRecorder().run()