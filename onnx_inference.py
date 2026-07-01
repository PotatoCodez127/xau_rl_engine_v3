import numpy as np
import onnxruntime as ort

class LocalInferenceEngine:
    """
    Ultra-low latency inference engine running purely on the local i5 CPU.
    No PyTorch or SB3 dependencies.
    """
    def __init__(self, oracle_onnx_path: str, manager_onnx_path: str):
        print("Initializing Local ONNX Runtime Environment (i5 Optimized)...")
        
        # Configure ONNX to maximize local CPU thread efficiency
        sess_options = ort.SessionOptions()
        # Restrict to 2-4 threads to leave room for the FastAPI event loop & MT5
        sess_options.intra_op_num_threads = 2 
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load Execution Sessions natively on CPU
        self.oracle_session = ort.InferenceSession(oracle_onnx_path, sess_options, providers=['CPUExecutionProvider'])
        self.manager_session = ort.InferenceSession(manager_onnx_path, sess_options, providers=['CPUExecutionProvider'])

        self.oracle_input_name = self.oracle_session.get_inputs()[0].name
        self.manager_input_name = self.manager_session.get_inputs()[0].name

    def softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def predict_oracle(self, sequence_buffer: np.ndarray) -> tuple[float, float, float]:
        """Runs the Temporal Attention Phase A Oracle."""
        input_tensor = np.expand_dims(sequence_buffer, axis=0).astype(np.float32)
        logits = self.oracle_session.run(None, {self.oracle_input_name: input_tensor})[0]
        probs = self.softmax(logits)[0]
        return float(probs[0]), float(probs[1]), float(probs[2])

    def predict_manager(self, obs_vector: np.ndarray) -> tuple[float, float]:
        """Runs the Distributional SAC Phase B Actor."""
        input_tensor = np.expand_dims(obs_vector, axis=0).astype(np.float32)
        actions = self.manager_session.run(None, {self.manager_input_name: input_tensor})[0]
        action_array = actions[0]
        return float(action_array[0]), float(action_array[1])