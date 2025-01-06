from kubernetes import client, config

# Load the kubeconfig file from a specific path
kubeconfig_path = "./kubeconfig"
config.load_kube_config(config_file=kubeconfig_path)

def scale_deployment(namespace: str, deployment_name: str, replicas: int):
    """
    Scale the replicas of a deployment.
    
    :param namespace: The namespace of the deployment.
    :param deployment_name: The name of the deployment.
    :param replicas: The desired number of replicas.
    """

    # Create an AppsV1Api client
    apps_v1 = client.AppsV1Api()

    # Fetch the current deployment to get its full specification
    deployment = apps_v1.read_namespaced_deployment(name=deployment_name, namespace=namespace)

    # Update the number of replicas
    deployment.spec.replicas = replicas

    # Patch the deployment with the updated replicas
    apps_v1.patch_namespaced_deployment(
        name=deployment_name,
        namespace=namespace,
        body=deployment
    )
    print(f"Scaled deployment '{deployment_name}' in namespace '{namespace}' to {replicas} replicas.")

def get_pod_count(namespace: str, deployment_name: str) -> int:
    """
    Get the count of pods for a specific deployment.
    
    :param namespace: The namespace of the deployment.
    :param deployment_name: The name of the deployment.
    :return: The number of running pods managed by the deployment.
    """

    # Create an AppsV1Api client
    apps_v1 = client.AppsV1Api()

    # Fetch the deployment details
    deployment = apps_v1.read_namespaced_deployment(name=deployment_name, namespace=namespace)

    # Create a CoreV1Api client to list pods
    core_v1 = client.CoreV1Api()

    # List all pods in the namespace with a label selector matching the deployment's labels
    label_selector = ",".join([f"{k}={v}" for k, v in deployment.spec.selector.match_labels.items()])
    pods = core_v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)

    # Filter pods based on their status to count only running pods
    running_pods = [pod for pod in pods.items if pod.status.phase == "Running"]

    # Print details
    print(f"Deployment '{deployment_name}' in namespace '{namespace}':")
    print(f"- Desired replicas: {deployment.spec.replicas}")
    print(f"- Running pods: {len(running_pods)}")

    return len(running_pods)

# get_pod_count("app", "helloword-helloworld")
# scale_deployment("app", "helloword-helloworld", 3)
