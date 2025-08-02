import sys
print("Python path:")
for path in sys.path:
    print(f"  {path}")

print("\nTrying to import websocket...")
try:
    import websocket
    print(f"websocket module: {websocket}")
    print(f"websocket location: {websocket.__file__}")
    print(f"websocket contents: {dir(websocket)}")
except Exception as e:
    print(f"Error importing websocket: {e}")

print("\nTrying to import websocket-client...")
try:
    import websocket_client
    print(f"websocket_client module: {websocket_client}")
    print(f"websocket_client location: {websocket_client.__file__}")
except Exception as e:
    print(f"Error importing websocket_client: {e}") 