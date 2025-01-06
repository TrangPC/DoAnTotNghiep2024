import requests
import time

def get_requests_per_minute(prometheus_url, ingress_name):
    """
    Fetch the number of requests per minute for the last 10 minutes.

    Args:
        prometheus_url (str): URL of the Prometheus server.
        ingress_name (str): Name of the ingress to filter.

    Returns:
        list: An array of 10 values, each representing the requests per minute for one minute.
    """
    # PromQL query
    query = f'sum(rate(nginx_ingress_controller_requests{{ingress="{ingress_name}"}}[1m])) by (ingress) * 60'
    end_time = int(time.time())  # Current Unix timestamp
    start_time = end_time - 600  # 10 minutes ago
    step = 60  # 1 minute step

    # Query Prometheus for a range of data
    response = requests.get(
        f"{prometheus_url}/api/v1/query_range",
        params={
            "query": query,
            "start": start_time,
            "end": end_time,
            "step": step,
        },
    )

    # Check response status
    if response.status_code != 200:
        raise Exception(f"Error querying Prometheus: {response.status_code}, {response.text}")

    data = response.json()
    results = data.get("data", {}).get("result", [])
    
    # Extract values
    values = []
    for result in results:
        for value in result.get("values", []):
            timestamp, count_per_minute = value
            values.append(float(count_per_minute))

    # Return the last 10 values
    return values

# prometheus_url = "http://10.148.0.2:30090"
# ingress_name = "helloword-helloworld"
# try:
#     request_counts = get_requests_per_minute(prometheus_url, ingress_name)
#     print(request_counts)
# except Exception as e:
#     print(f"Error: {e}")
