import os
import json
import subprocess
import re
import sys
import argparse
from datetime import datetime, timedelta
import logging

from openrouter_api import OpenRouterAPI

logger = logging.getLogger(__name__)

def load_config(config_path='config.json'):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        required_keys = ["models", "max_soundings_per_day", "logging_enabled", 
                         "data_dir", "start_date", "end_date", "api_key_file"]
        if not all(key in config for key in required_keys):
            raise ValueError("Config file missing required keys.")
        if not isinstance(config['models'], list) or len(config['models']) == 0:
            raise ValueError("'models' must be a non-empty array of model names.")
        if 'model_name' in config and 'models' not in config:
            config['models'] = [config['model_name']]
        datetime.strptime(config['start_date'], "%Y%m%d")
        datetime.strptime(config['end_date'], "%Y%m%d")
        if not isinstance(config['max_soundings_per_day'], int) or \
           not isinstance(config['logging_enabled'], bool):
           raise ValueError("Invalid type for max_soundings_per_day or logging_enabled in config.")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not parse configuration file {config_path}. Check JSON format.", file=sys.stderr)
        sys.exit(1)
    except (ValueError, KeyError) as e:
        print(f"Error in configuration file {config_path}: {e}", file=sys.stderr)
        sys.exit(1)

def setup_logging(config):
    global logger
    if config.get("logging_enabled", False):
        LOGS_DIR = "logs"
        os.makedirs(LOGS_DIR, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("AgentCaster")
        logger.info("Logging enabled.")
    else:
        logging.disable(logging.CRITICAL)
        logger.disabled = True
        print("Logging is disabled in the configuration.")

def setup_daily_logger(date_str, config, model_name=None):
    if not config.get("logging_enabled", False):
        return None
        
    LOGS_DIR = "logs"
    model_to_use = model_name if model_name else config.get("model_name", "unknown_model")
    model_name_sanitized = model_to_use.replace("/", "_")
    
    max_soundings_per_day = config.get("max_soundings_per_day", "unknown")
    filename_suffix = f"_soundings_{max_soundings_per_day}"
    
    log_filename = os.path.join(LOGS_DIR, f"agent_interaction_{model_name_sanitized}{filename_suffix}_{date_str}.log")
    daily_logger = logging.getLogger(f"{model_name_sanitized}_{max_soundings_per_day}_{date_str}")
    daily_logger.setLevel(logging.INFO)
    
    if not daily_logger.handlers:
        fh = logging.FileHandler(log_filename, mode='w')
        fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)
        daily_logger.addHandler(fh)
        
    daily_logger.propagate = False 
    return daily_logger

def get_logger(date_str=None):
    if date_str:
        daily_logger = logging.getLogger(date_str)
        return daily_logger if daily_logger.hasHandlers() else logger
    return logger

def scrub_image_data(content):
    if isinstance(content, list):
        scrubbed_content = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "image_url" and part.get("image_url", {}).get("url", "").startswith("data:image"):
                    img_path = part.get("image_url", {}).get("source_path", "unknown_path") 
                    scrubbed_content.append({**part, "image_url": {"url": f"[Image data removed for log: {img_path}]"}})
                else:
                    scrubbed_content.append(part)
            else:
                 scrubbed_content.append(part)
        return scrubbed_content
    elif isinstance(content, str):
        if content.startswith("data:image"):
             return "[Image data removed for log]"
        return content
    elif isinstance(content, dict):
        return {k: scrub_image_data(v) for k, v in content.items()}
    else:
        return content

def get_tools(config):
    MAX_SOUNDINGS_PER_DAY = config["max_soundings_per_day"]

    LIST_MAP_TYPES_TOOL = {
        "type": "function",
        "function": {
            "name": "list_available_map_types",
            "description": "Lists the available types of HRRR map plots based on the generated directories. Call this first to see what map types can be requested.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }

    REQUEST_HRRR_MAP_TOOL = {
        "type": "function",
        "function": {
            "name": "request_hrrr_map",
            "description": "Requests a specific HRRR forecast map image (PNG). Provide the exact map type directory name (obtained from list_available_map_types) and the forecast hour.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map_type_directory": {
                        "type": "string",
                        "description": "The exact directory name representing the map type (e.g., 'Relative_humidity_at_0_isothermZero', 'Convective_available_potential_energy_at_0_surface'). Obtain this from list_available_map_types."
                    },
                    "forecast_hour": {
                        "type": "integer",
                        "description": "The forecast hour (e.g., 12, 18, 36) for the map."
                    }
                },
                "required": ["map_type_directory", "forecast_hour"]
            }
        }
    }

    REQUEST_SOUNDING_TOOL = {
        "type": "function",
        "function": {
            "name": "request_sounding",
            "description": f"Requests a forecast sounding plot (PNG) for the nearest available station to the specified location and forecast hour. Limit of {MAX_SOUNDINGS_PER_DAY} soundings per day.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Target latitude in decimal degrees."
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Target longitude in decimal degrees."
                    },
                    "forecast_hour": {
                        "type": "integer",
                        "description": "The forecast hour for the sounding (e.g., 12, 15, 24)."
                    }
                },
                "required": ["latitude", "longitude", "forecast_hour"]
            }
        }
    }

    SUBMIT_TORNADO_PREDICTION_TOOL = {
        "type": "function",
        "function": {
            "name": "submit_tornado_prediction",
            "description": "Call this function ONLY when you have finished analyzing all necessary maps and soundings and are ready to submit the final tornado risk prediction as GeoJSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prediction_geojson": {
                        "type": "string",
                        "description": "Output Requirements: A valid GeoJSON FeatureCollection string representing the tornado risk forecast. \n- Each Feature must be a Polygon or MultiPolygon for a single risk category (2%, 5%, 10%, 15%, 30%, 45%, 60%). \n- Each Feature MUST have a 'properties' field with a 'risk_level' key holding the percentage string (e.g., {'risk_level': '5%'}). \n- The <2% area is implicitly defined by the forecast domain not covered by higher risks; DO NOT explicitly include a <2% Feature. \n- CRITICAL NESTING: Higher risk polygons MUST be spatially contained within ALL lower risk polygons (e.g., 10% inside 5%, 5% inside 2%). \n- Use MultiPolygon geometry for a single Feature if a risk level covers multiple disjoint areas."
                    }
                },
                "required": ["prediction_geojson"]
            }
        }
    }
    
    return [LIST_MAP_TYPES_TOOL, REQUEST_HRRR_MAP_TOOL, REQUEST_SOUNDING_TOOL, SUBMIT_TORNADO_PREDICTION_TOOL]

def list_map_type_dirs(date_str, data_dir):
    maps_base_dir = os.path.join(data_dir, f"hrrr_{date_str}_00z", "hrrr_plots")
    available_types = []
    if not os.path.isdir(maps_base_dir):
        get_logger(date_str).warning(f"HRRR plots directory not found: {maps_base_dir}")
        return []
    try:
        for item in os.listdir(maps_base_dir):
            item_path = os.path.join(maps_base_dir, item)
            if os.path.isdir(item_path):
                available_types.append(item)
        return sorted(available_types)
    except Exception as e:
        get_logger(date_str).exception(f"Error listing map type directories in {maps_base_dir}")
        return []

def execute_sounding_script(date_str, lat, lon, fcst_hour, data_dir):
    script_path = "find_and_plot_nearest_sounding.py"
    command = [
        sys.executable,
        script_path,
        date_str,
        str(lat),
        str(lon),
        str(fcst_hour),
        "--data_dir", data_dir
    ]

    current_logger = get_logger(date_str)
    current_logger.info(f"Executing command: {' '.join(command)}")

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, cwd=os.path.dirname(os.path.abspath(__file__)))

        if current_logger and not current_logger.disabled:
            current_logger.info(f"--- {script_path} stdout ---")
            if result.stdout: current_logger.info(result.stdout.strip())
            current_logger.info(f"--- {script_path} stderr ---")
            if result.stderr: current_logger.info(result.stderr.strip()) 
            current_logger.info(f"--- End {script_path} output (Return Code: {result.returncode}) ---")

        if result.returncode == 0:
            match = re.search(r"PNG Path:\s*(.*\.png)", result.stdout)
            if match:
                png_path = match.group(1).strip()
                if os.path.exists(png_path):
                    return {"success": True, "png_path": png_path}
                else:
                    current_logger.error(f"Script succeeded but PNG path not found or invalid: {png_path}")
                    return {"success": False, "error": f"Script succeeded but output PNG path not found: {png_path}"}
            else:
                current_logger.error(f"Script succeeded but could not parse PNG path from output.")
                return {"success": False, "error": "Script succeeded but could not parse PNG path from stdout."}
        else:
            error_message = f"{script_path} failed with return code {result.returncode}."
            if "No BUFKIT files found" in result.stderr:
                error_message += " (No BUFKIT files found for the date)"
            elif "No profile found for forecast hour" in result.stderr:
                 error_message += f" (Forecast hour {fcst_hour} not available in the selected BUFKIT file)"
            elif "BUFKIT to SHARPpy conversion failed" in result.stderr:
                 error_message += " (BUFKIT to SHARPpy conversion failed)"
                 
            current_logger.error(error_message)
            stderr_snippet = (result.stderr[:200] + '...') if len(result.stderr) > 200 else result.stderr
            return {"success": False, "error": error_message, "details": stderr_snippet.strip()}

    except FileNotFoundError:
        get_logger(date_str).exception(f"Error: Could not execute Python interpreter '{sys.executable}' or script '{script_path}'")
        return {"success": False, "error": f"Could not find or execute {script_path}."}
    except Exception as e:
        get_logger(date_str).exception(f"An unexpected error occurred running {script_path}: {e}")
        return {"success": False, "error": f"Unexpected error running sounding script: {str(e)}"}

def run_agent_interaction(config, model_name):
    start_date = datetime.strptime(config["start_date"], "%Y%m%d")
    end_date = datetime.strptime(config["end_date"], "%Y%m%d")
    current_date = start_date
    data_dir = config["data_dir"]
    api_key_file = config["api_key_file"]
    max_soundings_per_day = config["max_soundings_per_day"]
    tools = get_tools(config)

    if not os.path.exists(api_key_file):
        get_logger().error(f"API key file not found: {api_key_file}")
        sys.exit(1)
    if not os.path.isdir(data_dir):
         get_logger().error(f"Data directory not found: {data_dir}")
         get_logger().error("Please ensure the dataset has been downloaded and organized correctly.")
         sys.exit(1)

    openrouter = OpenRouterAPI(api_key_file=api_key_file)
    openrouter.model = model_name

    get_logger().info(f"Starting agent interaction for model: {model_name}")

    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        daily_logger = setup_daily_logger(date_str, config, model_name) 
        current_logger = daily_logger if daily_logger else get_logger() 
        
        current_logger.info(f"--- Starting Agent Interaction for Date: {date_str} with Model: {model_name} ---")

        messages = []
        soundings_requested_today = 0

        lambda_latex = "\\( \\lambda \\)"
        prob_formula_latex = f"\\( P = 1 - e^{{{{-\\lambda}}}} \\)"
        system_prompt = f"""
**You are AgentCaster, an expert AI meteorologist agent that advises the Storm Prediction Center (SPC) in tornado prediction using 00z HRRR model data.**

**Objective:**
Your primary objective is to utilize HRRR forecast data to generate an SPC-style tornado risk forecast for the CONUS for the forecast day starting {date_str} 12z to {current_date + timedelta(days=1):%Y%m%d} 12z (forecast hours 12-36 from the 00z run). This is the timeframe for which you will be making your SPC-style prediction.

**Background & Evaluation:**
To evaluate your prediction, the ground truth is generated as follows: Observed tornado reports are used to calculate a normalized probability density field on an ~80km grid (using a Gaussian kernel with \( \sigma \approx 120 \text{{km}} \)), which is then interpolated to a ~5km grid. This density field is convolved with a 40km radius disk kernel to integrate the density over a neighborhood. The result is multiplied by the grid cell area to get an expected tornado count ({lambda_latex}). Finally, this expected count is converted to a probability using {prob_formula_latex}. This probability field is categorized using standard SPC thresholds (2%, 5%, 10%, etc.) and converted into vector polygon geometries. Your predicted risk areas (from the GeoJSON you provide) are directly compared against these ground truth geometries using vector-based geometric operations. Your final score is the average Intersection over Union (IoU) across all evaluated categories present in either your prediction or the ground truth, calculated based on the areas of the geometric intersection and union. This score ranges from 0% (no agreement) to 100% (perfect agreement). Accurate placement, spatial extent, and correct nesting of risk levels (2%, 5%, 10%, etc.) are crucial for a high score. The tornado risk probabilities you predict (e.g., 5%, 10%) represent the likelihood of a tornado occurring within 25 miles (approx. 40 km) of any point within that specific risk area during the forecast period ({date_str} 12z to {current_date + timedelta(days=1):%Y%m%d} 12z).

**Data & Tools:**
You can request HRRR map plots and sounding diagrams using the following available tools:
1.  `list_available_map_types`: Call this first to see the list of available map types (represented by directory names).
2.  `request_hrrr_map`: Gets a specific map plot (PNG). Provide the exact `map_type_directory` name from the list and the integer `forecast_hour` (12-36).
3.  `request_sounding`: Gets a sounding plot (PNG) for the nearest available station to a specified lat/lon for a specific integer `forecast_hour` (12-36). You are limited to {max_soundings_per_day} sounding requests per day.
4.  `submit_tornado_prediction`: Call this function ONLY ONCE when you have finished analyzing all necessary maps and soundings and are ready to submit the final tornado risk prediction. You MUST provide the prediction as a valid GeoJSON FeatureCollection string in the `prediction_geojson` argument. See the tool description for detailed GeoJSON format requirements.

**Workflow Guidance:**
- Start by calling `list_available_map_types` to understand the data available for today.
- Then, use `request_hrrr_map` and `request_sounding` (strategically, respecting the quota) to gather the information needed for your analysis.
- When confident, call the `submit_tornado_prediction` function with the properly formatted and nested GeoJSON output, ensuring all separate areas for each risk level are included.
"""
        messages.append({"role": "system", "content": system_prompt})
        current_logger.info(f"System: {system_prompt}")

        initial_user_prompt = f"""Today's forecast date is {date_str}.
You have {max_soundings_per_day} sounding requests available for today.
Please start by calling `list_available_map_types` to see the available map plots. Remember to call `submit_tornado_prediction` with your final GeoJSON prediction when you are confident with your analysis."""
        messages.append({"role": "user", "content": initial_user_prompt})
        current_logger.info(f"User: {initial_user_prompt}")

        MAX_CONSECUTIVE_ERRORS = 3
        consecutive_errors = 0
        interaction_complete = False
        while not interaction_complete:
            current_logger.info(f"Requesting completion from OpenRouter using model: {model_name}")
            try:
                response = openrouter.chat_completion(messages=messages, tools=tools, tool_choice="auto")
                if config.get("logging_enabled", False):
                    current_logger.info(f"Raw API Response: {json.dumps(scrub_image_data(response), indent=2)}") 
                consecutive_errors = 0
            except Exception as e:
                current_logger.exception(f"Error calling OpenRouter API with model {model_name}.")
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                     current_logger.error(f"Max consecutive API errors ({MAX_CONSECUTIVE_ERRORS}) reached. Skipping day {date_str}.")
                     break
                else:
                     current_logger.warning(f"API error occurred ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}). Retrying...")
                     continue

            if not response or "choices" not in response or not response["choices"]:
                 current_logger.error(f"Received empty or invalid response from API with model {model_name}.")
                 current_logger.error("Skipping rest of interaction for this day.")
                 break
            
            assistant_message = response["choices"][0].get("message")

            if not assistant_message:
                 current_logger.error(f"API response missing 'message' field for model {model_name}.")
                 current_logger.error("Skipping rest of interaction for this day.")
                 break

            messages.append(assistant_message)
            if config.get("logging_enabled", False):
                 current_logger.info(f"Assistant: {json.dumps(scrub_image_data(assistant_message), indent=2)}")

            if assistant_message.get("tool_calls"):
                tool_calls = assistant_message["tool_calls"]
                tool_messages_to_append = []
                all_image_paths_this_turn = []
                soundings_requested_this_turn = False

                for tool_call in tool_calls:
                    function_call = tool_call.get("function", {})
                    tool_name = function_call.get("name")
                    tool_id = tool_call.get("id")
                    
                    current_logger.info(f"Executing Tool Call: ID={tool_id}, Name={tool_name}")

                    if tool_name == "submit_tornado_prediction":
                        current_logger.info(f"LLM model {model_name} called submit_tornado_prediction.")
                        prediction_saved = False
                        try:
                            tool_args = json.loads(function_call.get("arguments", "{}"))
                            prediction_geojson_str = tool_args.get("prediction_geojson")

                            if prediction_geojson_str:
                                try:
                                    prediction_data = json.loads(prediction_geojson_str)
                                    current_logger.info(f"Received Prediction GeoJSON (raw string): {prediction_geojson_str}")
                                    current_logger.info(f"Received Prediction GeoJSON (parsed): {json.dumps(prediction_data, indent=2)}") 
                                    
                                    try:
                                        output_dir = os.path.join("llm_predictions", date_str)
                                        os.makedirs(output_dir, exist_ok=True)
                                        
                                        model_name_sanitized = model_name.replace("/", "_")
                                        
                                        filename_suffix = f"_soundings_{max_soundings_per_day}"
                                            
                                        output_filename = f"prediction_{model_name_sanitized}{filename_suffix}_{date_str}.geojson"
                                        output_path = os.path.join(output_dir, output_filename)
                                        
                                        with open(output_path, 'w') as f:
                                            json.dump(prediction_data, f, indent=2)
                                        current_logger.info(f"Successfully saved prediction GeoJSON to: {output_path}")
                                        prediction_saved = True
                                    except Exception as save_err:
                                        current_logger.exception(f"Error saving prediction GeoJSON to file: {save_err}")
                                    
                                    if prediction_saved:
                                        tool_result_content = json.dumps({"success": True, "message": f"Prediction GeoJSON received and saved to {output_path}."})
                                    else:
                                        tool_result_content = json.dumps({"success": True, "message": "Prediction GeoJSON received, but failed to save to file (see logs)."})
                                except json.JSONDecodeError as json_err:
                                    current_logger.error(f"Failed to parse prediction_geojson string as JSON: {json_err}")
                                    current_logger.error(f"Received string: {prediction_geojson_str}")
                                    tool_result_content = json.dumps({"success": False, "error": f"Invalid JSON provided in prediction_geojson: {json_err}"})
                            else:
                                current_logger.error("Tool call 'submit_tornado_prediction' missing required argument 'prediction_geojson'.")
                                tool_result_content = json.dumps({"success": False, "error": "Missing required argument: prediction_geojson."})
                        except json.JSONDecodeError as arg_err:
                             current_logger.error(f"Failed to decode arguments for tool {tool_name}: {function_call.get('arguments')}. Error: {arg_err}")
                             tool_result_content = json.dumps({"success": False, "error": "Invalid arguments format (not JSON)"})
                        except Exception as e:
                             current_logger.exception(f"Unexpected error processing tool {tool_name}")
                             tool_result_content = json.dumps({"success": False, "error": f"Unexpected error: {str(e)}"})

                        messages.append({"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": tool_result_content})
                        current_logger.info(f"Tool Result: ID={tool_id}, Content={tool_result_content}")
                        interaction_complete = True
                        break
                    
                    elif tool_name == "list_available_map_types":
                        map_dirs = list_map_type_dirs(date_str, data_dir)
                        if map_dirs:
                            tool_result_content = json.dumps({"success": True, "available_map_types": map_dirs})
                        else:
                             tool_result_content = json.dumps({"success": False, "error": "No map type directories found for this date."}) 
                        tool_messages_to_append.append({"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": tool_result_content})
                        current_logger.info(f"Tool Result: ID={tool_id}, Content={tool_result_content}")
                        continue

                    try:
                        tool_args = json.loads(function_call.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        current_logger.error(f"Failed to decode arguments for tool {tool_name}: {function_call.get('arguments')}")
                        tool_result_content = json.dumps({"success": False, "error": "Invalid arguments format (not JSON)."})
                        tool_messages_to_append.append({"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": tool_result_content})
                        continue 

                    tool_failed = False

                    if tool_name == "request_hrrr_map":
                        map_dir_name = tool_args.get("map_type_directory")
                        fh = tool_args.get("forecast_hour")
                        if map_dir_name and fh is not None:
                            map_type_path = os.path.join(data_dir, f"hrrr_{date_str}_00z", "hrrr_plots", map_dir_name)
                            
                            found_map_paths = []
                            if os.path.isdir(map_type_path):
                                try:
                                    target_fh_str = f"fcf{fh}"
                                    for filename in os.listdir(map_type_path):
                                        if target_fh_str in filename and map_dir_name in filename and filename.endswith(".png"):
                                            found_map_paths.append(os.path.join(map_type_path, filename))
                                            
                                except Exception as e:
                                    current_logger.exception(f"Error listing or searching directory {map_type_path}")
                                    tool_failed = True
                                    tool_result_content = json.dumps({"success": False, "error": f"Error accessing map directory: {map_type_path}"})
                                    
                            else:
                                current_logger.warning(f"Map type directory not found: {map_type_path}")

                            if found_map_paths and not tool_failed:
                                found_basenames = [os.path.basename(p) for p in found_map_paths]
                                message = f"Found {len(found_map_paths)} map image(s) for {map_dir_name} F{fh}. Providing all found images."
                                message += f" Found files: {', '.join(found_basenames)}"
                                tool_result_content = json.dumps({ 
                                    "success": True, 
                                    "message": message,
                                    "found_files": found_basenames, 
                                    "image_paths": found_map_paths
                                })
                                all_image_paths_this_turn.extend(found_map_paths)
                            elif not tool_failed: 
                                tool_result_content = json.dumps({"success": False, "error": f"Requested map file not found for F{fh} in directory: {map_dir_name}. Searched path: {map_type_path}"})
                                tool_failed = True
                                current_logger.warning(f"Tool Failure: {tool_name} - Map PNG not found matching F{fh} in directory: {map_type_path}")
                        else:
                            tool_result_content = json.dumps({"success": False, "error": "Missing required arguments: map_type_directory or forecast_hour."}) 
                            tool_failed = True
                            current_logger.warning(f"Tool Failure: {tool_name} - Missing arguments.")
                            
                    elif tool_name == "request_sounding":
                        lat = tool_args.get("latitude")
                        lon = tool_args.get("longitude")
                        fh = tool_args.get("forecast_hour")

                        if lat is not None and lon is not None and fh is not None:
                            if soundings_requested_today >= max_soundings_per_day:
                                tool_result_content = json.dumps({"success": False, "error": f"Sounding request limit ({max_soundings_per_day}) reached for today."})
                                tool_failed = True
                                current_logger.warning(f"Tool Failure: {tool_name} - Sounding limit reached.")
                            else:
                                script_result = execute_sounding_script(date_str, lat, lon, fh, data_dir)
                                if script_result and script_result.get("success"):
                                    soundings_requested_today += 1
                                    remaining = max_soundings_per_day - soundings_requested_today
                                    image_path = script_result.get("png_path")
                                    message = f"Sounding plot generated for nearest station to ({lat}, {lon}) F{fh}. Image provided."
                                    tool_result_content = json.dumps({
                                         "success": True, 
                                         "message": message, 
                                         "image_path": image_path,
                                         "soundings_remaining": remaining
                                    })
                                    if image_path:
                                         all_image_paths_this_turn.append(image_path)
                                         soundings_requested_this_turn = True
                                else:
                                    error_msg = script_result.get("error", "Unknown error running sounding script.")
                                    tool_result_content = json.dumps({"success": False, "error": error_msg})
                                    tool_failed = True
                                    current_logger.warning(f"Tool Failure: {tool_name} - Sounding script failed: {error_msg}")
                        else:
                           tool_result_content = json.dumps({"success": False, "error": "Missing required arguments: latitude, longitude, or forecast_hour."}) 
                           tool_failed = True
                           current_logger.warning(f"Tool Failure: {tool_name} - Missing arguments.")
                    
                    else:
                        tool_result_content = json.dumps({"success": False, "error": f"Unknown tool name: {tool_name}"})
                        tool_failed = True
                        current_logger.warning(f"Tool Failure: Unknown tool '{tool_name}' called.")

                    current_logger.info(f"Tool Result: ID={tool_id}, Content={tool_result_content}")
                    tool_messages_to_append.append({"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": tool_result_content})

                prepared_image_message_content = None
                image_path_for_logging = []

                if all_image_paths_this_turn:
                    image_paths = all_image_paths_this_turn
                    image_path_for_logging = image_paths

                    prepared_content_parts = []
                    all_images_encoded = True
                    error_files = []

                    text_part = "Here is the requested data:"
                    if soundings_requested_this_turn:
                         remaining = max_soundings_per_day - soundings_requested_today
                         quota_text = f" You have {remaining} sounding requests remaining for today."
                         text_part += quota_text
                    prepared_content_parts.append({"type": "text", "text": text_part})

                    for img_path in image_paths:
                         try:
                            prepared_content_parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{openrouter.encode_image(img_path)}",
                                        "source_path": img_path
                                    }
                                }
                            )
                         except FileNotFoundError:
                            current_logger.error(f"Image file not found when preparing user message: {img_path}")
                            all_images_encoded = False
                            error_files.append(os.path.basename(img_path))
                         except Exception as e:
                            current_logger.exception(f"Error encoding or adding image {img_path}")
                            all_images_encoded = False
                            error_files.append(os.path.basename(img_path))

                    if all_images_encoded:
                         prepared_image_message_content = prepared_content_parts
                    else:
                         error_text = f"(System Note: An error occurred trying to attach the image(s) for: {', '.join(error_files)}. Showing any successfully attached images.) What would you like to do next?"
                         prepared_content_parts.append({"type": "text", "text": error_text})
                         prepared_image_message_content = prepared_content_parts
                         image_path_for_logging = []

                if not interaction_complete:
                    messages.extend(tool_messages_to_append)
                
                if interaction_complete:
                    break
                    
                if prepared_image_message_content:
                     user_message_with_image = {"role": "user", "content": prepared_image_message_content}
                     messages.append(user_message_with_image)
                     if config.get("logging_enabled", False):
                        logged_message = {"role": "user"}
                        if isinstance(prepared_image_message_content, list):
                            logged_message["content"] = scrub_image_data({"content": prepared_image_message_content})["content"]
                        else:
                            logged_message["content"] = prepared_image_message_content
                        
                        if image_path_for_logging:
                             logged_message["image_paths_attempted"] = image_path_for_logging
                             
                        current_logger.info(f"User (Image/Note): {json.dumps(logged_message, indent=2)}")
                     
                
                continue

            elif assistant_message.get("content"):
                 consecutive_errors = 0
                 pass

            else:
                 current_logger.error("Assistant message exists but is missing 'content' and 'tool_calls'.")
                 consecutive_errors += 1
                 if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                     current_logger.error(f"Max consecutive API errors/invalid responses ({MAX_CONSECUTIVE_ERRORS}) reached. Skipping day {date_str}.")
                     break
                 else:
                     current_logger.warning(f"Invalid assistant message ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}). Retrying API call or breaking depending on configuration...")
                     break

        current_logger.info(f"--- Ending Agent Interaction for Date: {date_str} with Model: {model_name} ---")
        current_date += timedelta(days=1)

def run_multiple_models(config):
    models = config["models"]
    if not models:
        get_logger().error("No models specified in config.")
        return
        
    get_logger().info(f"Starting multi-model run with {len(models)} models: {', '.join(models)}")
    
    for model_name in models:
        get_logger().info(f"=== Starting run for model: {model_name} ===")
        run_agent_interaction(config, model_name)
        get_logger().info(f"=== Completed run for model: {model_name} ===")
    
    get_logger().info(f"Multi-model run completed for all {len(models)} models.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AgentCaster LLM interaction for tornado forecasting.")
    parser.add_argument("--config", default="config.json", help="Path to the JSON configuration file (default: config.json)")
    parser.add_argument("--start_date", help="Override start date in YYYYMMDD format (optional)")
    parser.add_argument("--end_date", help="Override end date in YYYYMMDD format (optional)")
    parser.add_argument("--model", help="Run only a specific model from the models list (optional)")

    args = parser.parse_args()

    config = load_config(args.config)
    
    if args.start_date:
        try:
            datetime.strptime(args.start_date, "%Y%m%d")
            config["start_date"] = args.start_date
            print(f"Overriding start date with: {args.start_date}")
        except ValueError:
             print(f"Error: Invalid override start_date format: {args.start_date}. Using config value.", file=sys.stderr)
    if args.end_date:
        try:
            datetime.strptime(args.end_date, "%Y%m%d")
            config["end_date"] = args.end_date
            print(f"Overriding end date with: {args.end_date}")
        except ValueError:
             print(f"Error: Invalid override end_date format: {args.end_date}. Using config value.", file=sys.stderr)

    setup_logging(config)
    
    if args.model:
        if args.model in config["models"]:
            get_logger().info(f"Running only model: {args.model}")
            run_agent_interaction(config, args.model)
        else:
            get_logger().error(f"Specified model '{args.model}' not found in config's models list.")
            sys.exit(1)
    else:
        run_multiple_models(config)

    get_logger().info("Agent interaction process completed.")
    print("\nAgent interaction process completed.") 