"""Testler gerçek logs/ ve reports/ klasörlerini kirletmesin."""
import os
import tempfile

os.environ["BOT_LOG_DIR"] = tempfile.mkdtemp(prefix="bot_test_logs_")
os.environ["BOT_REPORT_DIR"] = tempfile.mkdtemp(prefix="bot_test_reports_")
