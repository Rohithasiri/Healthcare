"""Test imports for rPPG service"""
import sys

print("Testing imports...")

try:
    from rppg_processor import RPPGProcessor
    print("✅ Primary processor (PhysNet) - OK")
except Exception as e:
    print(f"❌ Primary processor - Error: {e}")

try:
    from backup_processor import BackupRPPGProcessor
    print("✅ Backup processor - OK")
except Exception as e:
    print(f"❌ Backup processor - Error: {e}")
