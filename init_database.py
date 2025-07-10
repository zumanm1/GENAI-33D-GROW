#!/usr/bin/env python3
"""
Database Initialization Script
=============================

This script initializes the SQLite database for the GenAI Network Automation feature.
It creates the devices table and adds sample devices for testing.
"""

import sqlite3
import os

def init_devices_database():
    """Initialize the devices database with sample data."""
    
    # Database file path
    db_path = "devices.db"
    
    try:
        # Connect to database (creates if doesn't exist)
        with sqlite3.connect(db_path) as conn:
            # Create devices table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS devices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    host TEXT NOT NULL,
                    username TEXT NOT NULL,
                    password TEXT NOT NULL,
                    device_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Check if we have any devices
            cursor = conn.execute("SELECT COUNT(*) FROM devices")
            count = cursor.fetchone()[0]
            
            if count == 0:
                # Add sample devices
                sample_devices = [
                    ("R15", "172.16.39.102", "admin", "cisco", "cisco_ios_telnet"),
                    ("R16", "172.16.39.103", "admin", "cisco", "cisco_ios_telnet"),
                    ("R17", "172.16.39.104", "admin", "cisco", "cisco_ios_telnet"),
                    ("R18", "172.16.39.105", "admin", "cisco", "cisco_ios_telnet"),
                    ("R19", "172.16.39.106", "admin", "cisco", "cisco_ios_telnet"),
                    ("R20", "172.16.39.107", "admin", "cisco", "cisco_ios_telnet"),
                    ("Core-Switch-1", "192.168.1.1", "admin", "cisco123", "cisco_ios"),
                    ("Access-Switch-1", "192.168.1.2", "admin", "cisco123", "cisco_ios"),
                    ("Router-Edge", "10.0.0.1", "admin", "cisco123", "cisco_ios"),
                ]
                
                for device in sample_devices:
                    try:
                        conn.execute(
                            "INSERT INTO devices (name, host, username, password, device_type) VALUES (?, ?, ?, ?, ?)",
                            device
                        )
                        print(f"✅ Added device: {device[0]} ({device[1]})")
                    except sqlite3.IntegrityError:
                        print(f"⚠️ Device {device[0]} already exists")
                
                conn.commit()
                print(f"\n🎉 Database initialized with {len(sample_devices)} sample devices")
            else:
                print(f"✅ Database already contains {count} devices")
            
            # Show all devices
            print("\n📋 Current devices in database:")
            cursor = conn.execute("SELECT id, name, host, device_type FROM devices ORDER BY name")
            for row in cursor.fetchall():
                print(f"  {row[0]}. {row[1]} ({row[2]}) - {row[3]}")
                
    except Exception as e:
        print(f"❌ Database initialization failed: {str(e)}")
        return False
    
    return True

def add_device():
    """Interactive function to add a new device."""
    print("\n🔧 Add New Device")
    print("=" * 30)
    
    name = input("Device name: ").strip()
    host = input("IP address: ").strip()
    username = input("Username: ").strip()
    password = input("Password: ").strip()
    device_type = input("Device type (cisco_ios/cisco_ios_telnet): ").strip()
    
    if not all([name, host, username, password, device_type]):
        print("❌ All fields are required")
        return False
    
    try:
        with sqlite3.connect("devices.db") as conn:
            conn.execute(
                "INSERT INTO devices (name, host, username, password, device_type) VALUES (?, ?, ?, ?, ?)",
                (name, host, username, password, device_type)
            )
            conn.commit()
            print(f"✅ Device '{name}' added successfully")
            return True
    except sqlite3.IntegrityError:
        print(f"❌ Device '{name}' already exists")
        return False
    except Exception as e:
        print(f"❌ Error adding device: {str(e)}")
        return False

def main():
    """Main function to run the database initialization."""
    print("🤖 GenAI Network Automation - Database Initialization")
    print("=" * 55)
    
    # Initialize database
    if init_devices_database():
        print("\n✅ Database initialization completed successfully!")
        
        # Ask if user wants to add more devices
        while True:
            choice = input("\nWould you like to add another device? (y/n): ").lower().strip()
            if choice in ['y', 'yes']:
                add_device()
            elif choice in ['n', 'no']:
                break
            else:
                print("Please enter 'y' or 'n'")
        
        print("\n🎉 Database setup complete! You can now use the GenAI Network Automation feature.")
    else:
        print("\n❌ Database initialization failed!")

if __name__ == "__main__":
    main() 