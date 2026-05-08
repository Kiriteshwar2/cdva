import { useEffect, useState } from "react";

import { fetchMe, getStoredToken, login, setAccessToken, signup } from "../services/api";

export function useSession() {
  const [status, setStatus] = useState(() => (getStoredToken() ? "loading" : "anonymous"));
  const [user, setUser] = useState(null);

  useEffect(() => {
    const token = getStoredToken();
    if (!token) {
      return;
    }

    setAccessToken(token);
    fetchMe()
      .then((profile) => {
        setUser(profile);
        setStatus("authenticated");
      })
      .catch(() => {
        setAccessToken(null);
        setUser(null);
        setStatus("anonymous");
      });
  }, []);

  async function authenticate(mode, payload) {
    const action = mode === "signup" ? signup : login;
    const response = await action(payload);
    setAccessToken(response.access_token);
    setUser(response.user);
    setStatus("authenticated");
    return response.user;
  }

  function logout() {
    setAccessToken(null);
    setUser(null);
    setStatus("anonymous");
  }

  return {
    status,
    user,
    login: (payload) => authenticate("login", payload),
    signup: (payload) => authenticate("signup", payload),
    logout,
  };
}
