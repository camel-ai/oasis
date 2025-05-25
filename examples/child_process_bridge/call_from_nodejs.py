import asyncio
import inspect
import json
import sys

from reddit_user_generate import generate_user_data

async def handle_request(request):
  try:
    method = request['method']
    req_id = request['id']

    # retrieve function call in python file
    method_func = globals()[method]
    
    # check params
    sig = inspect.signature(method_func)
    param_count = len(sig.parameters)
    
    # call function with/without params
    if param_count == 0:
        result = await method_func()
    else:
        result = await method_func(request["params"])
    
    return {
      "id": req_id,
      "result": result,
      "error": None
    }
  except Exception as e:
    return {
      "id": req_id if 'id' in request else None,
      "result": None,
      "error": str(e)
    }

async def main():
  loop = asyncio.get_event_loop()
  
  # create async stdin reader
  reader = asyncio.StreamReader()
  protocol = asyncio.StreamReaderProtocol(reader)
  await loop.connect_read_pipe(lambda: protocol, sys.stdin)
  
  # listening stdin: function call with params
  while True:
    line = await reader.readline()
    if not line:
      break
        
    try:
      request = json.loads(line.decode())
      # create async task
      task = asyncio.create_task(handle_request(request))
      
      # callback with response
      def send_response(fut):
        response = fut.result()
        json_response = json.dumps(response) + "\n"
        sys.stdout.write(json_response)
        sys.stdout.flush()
          
      task.add_done_callback(send_response)
        
    except json.JSONDecodeError:
      error_response = json.dumps({
        "error": "Invalid JSON format"
      }) + "\n"
      sys.stdout.write(error_response)
      sys.stdout.flush()

if __name__ == "__main__":
  asyncio.run(main())