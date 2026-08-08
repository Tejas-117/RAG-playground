import axios from "axios";

/** The backend origin embedded into the browser bundle by Next.js. */
const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

// Fail during application setup instead of sending requests to an unintended origin.
if (!apiBaseUrl) {
  throw new Error("NEXT_PUBLIC_API_BASE_URL is not configured.");
}

/** Shared Axios client used for every request to the FastAPI backend. */
const apiClient = axios.create({
  baseURL: apiBaseUrl,
});

export default apiClient;
export { isAxiosError, isCancel } from "axios";
