import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
});

export async function fetchHealth() {
  const { data } = await client.get("/health");
  return data;
}

export async function generateStructures(numSamples) {
  const { data } = await client.post("/generate", { num_samples: numSamples });
  return data;
}

export { API_BASE_URL };
