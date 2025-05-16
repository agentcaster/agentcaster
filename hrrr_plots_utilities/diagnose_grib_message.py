import pygrib
import sys
import argparse

def list_all_messages(grib_file_path):
    print(f"--- Listing all messages in {grib_file_path} ---")
    try:
        grbs = pygrib.open(grib_file_path)
        count = 0
        for i, grb_message in enumerate(grbs):
            msg_num = i + 1
            try:
                name = grb_message.name
                short_name = grb_message.shortName
                level = grb_message.level
                type_of_level = grb_message.typeOfLevel
                print(f"  Msg {msg_num:>3}: Name='{name}', ShortName='{short_name}', Level={level}, TypeOfLevel='{type_of_level}'")
                count += 1
            except Exception as e:
                print(f"  Msg {msg_num:>3}: Error accessing basic keys - {e}")
                count += 1
        
        print(f"--- Found {count} total messages ---")
        grbs.seek(0)
        grbs.close()
        return True
        
    except FileNotFoundError:
        print(f"Error: GRIB file not found at {grib_file_path}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while listing messages: {e}")
        return False

def print_message_keys(grib_file_path, message_number):
    try:
        grbs = pygrib.open(grib_file_path)
        
        try:
            grb_message = grbs.message(message_number)
        except ValueError:
             print(f"Error: Message {message_number} not found by index in {grib_file_path}")
             grbs.close()
             return
        except Exception as e_idx:
             print(f"Error accessing message {message_number} by index: {e_idx}. Trying select...")
             grb_selected_list = grbs.select(message=message_number)
             if not grb_selected_list:
                 print(f"Error: Message {message_number} not found by select in {grib_file_path}")
                 grbs.close()
                 return
             grb_message = grb_selected_list[0]

        print(f"\n--- Detailed Metadata for Message {message_number} in {grib_file_path} ---")
        
        keys = grb_message.keys()
        
        keys.sort()
        
        for key in keys:
            try:
                value = grb_message[key]
                print(f"  {key}: {value}")
            except Exception as e:
                print(f"  {key}: Error accessing value - {e}")
                
        print("-" * (len(f"--- Detailed Metadata for Message {message_number} in {grib_file_path} ---")))
        
        grbs.close()

    except FileNotFoundError:
        print(f"Error: GRIB file not found at {grib_file_path}")
    except ValueError as e:
         print(f"Error processing message {message_number} in {grib_file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description='Diagnose GRIB message metadata by printing all keys.')
    parser.add_argument('grib_file', help='Path to the GRIB2 file to diagnose.')
    parser.add_argument('--msg1', type=int, default=8, help='First message number to inspect (default: 8)')
    parser.add_argument('--msg2', type=int, default=44, help='Second message number to inspect (default: 44)')

    args = parser.parse_args()

    print(f"Diagnosing GRIB File: {args.grib_file}")

    if list_all_messages(args.grib_file):
        print(f"\nComparing detailed metadata for Message: {args.msg1} and Message: {args.msg2}")
        print_message_keys(args.grib_file, args.msg1)
        print("\n")
        print_message_keys(args.grib_file, args.msg2)
    else:
        print("Could not list messages. Aborting detailed analysis.")

if __name__ == "__main__":
    main() 