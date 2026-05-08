import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const TOKEN_STORAGE_KEY = "cdvae_access_token";

let accessToken = null;

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 180000,
});

client.interceptors.request.use((config) => {
  if (accessToken) {
    config.headers.Authorization = `Bearer ${accessToken}`;
  }
  return config;
});

export function getStoredToken() {
  return window.localStorage.getItem(TOKEN_STORAGE_KEY);
}

export function setAccessToken(token) {
  accessToken = token || null;
  if (token) {
    window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
  } else {
    window.localStorage.removeItem(TOKEN_STORAGE_KEY);
  }
}

export async function signup(payload) {
  const { data } = await client.post("/auth/signup", payload);
  return data;
}

export async function login(payload) {
  const { data } = await client.post("/auth/login", payload);
  return data;
}

export async function fetchMe() {
  const { data } = await client.get("/auth/me");
  return data;
}

export async function fetchHealth() {
  const { data } = await client.get("/health");
  return data;
}

export async function fetchModels() {
  const { data } = await client.get("/models");
  return data;
}

export async function createGeneration(payload) {
  const { data } = await client.post("/generate", payload);
  
  // DEBUG: Verify CIF received from API
  console.log("\n%c[FRONTEND API] Generation Response Received", "color: #2563eb; font-weight: bold; font-size: 12px;");
  console.log("Response structure:", Object.keys(data));
  console.log("output_cif exists:", !!data.output_cif);
  console.log("output_cif type:", typeof data.output_cif);
  console.log("output_cif length:", data.output_cif?.length || "NOT FOUND");
  if (data.output_cif) {
    console.log("CIF preview:", data.output_cif.substring(0, 150));
    console.log("CIF valid format:", data.output_cif.startsWith("data_"));
  } else {
    console.warn("❌ WARNING: output_cif is MISSING from API response!");
    console.log("Available fields:", data);
  }
  console.log("%c" + "=".repeat(70), "color: #2563eb; font-size: 10px;");
  
  return data;
}

export async function fetchHistory(limit = 20) {
  const { data } = await client.get("/history", { params: { limit } });
  return data;
}

export async function fetchGeneration(id) {
  const { data } = await client.get(`/generation/${id}`);
  return data;
}

export { API_BASE_URL, TOKEN_STORAGE_KEY };
