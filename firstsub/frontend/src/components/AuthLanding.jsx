import { useMemo, useState } from "react";

import CrystalHero from "./CrystalHero";

const initialForms = {
  login: { email: "", password: "" },
  signup: { name: "", email: "", password: "" },
};

export default function AuthLanding({ onLogin, onSignup, busy, error }) {
  const [mode, setMode] = useState("login");
  const [forms, setForms] = useState(initialForms);

  const activeForm = forms[mode];
  const title = useMemo(
    () => (mode === "login" ? "Design the Future of Materials" : "Design the Future of Materials"),
    [mode],
  );

  function updateField(field, value) {
    setForms((current) => ({
      ...current,
      [mode]: {
        ...current[mode],
        [field]: value,
      },
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (mode === "login") {
      await onLogin(activeForm);
    } else {
      await onSignup(activeForm);
    }
  }

  return (
    <div className="landing-shell">
      <div className="landing-backdrop">
        <CrystalHero />
      </div>

      <section className="landing-hero">
        <div className="eyebrow">CDVAE Material Intelligence</div>
        <h1>{title}</h1>
        <p>AI-powered crystal generation with controllable physics</p>

      </section>

      <section className="auth-card">
        <div className="auth-switch">
          <button
            type="button"
            className={mode === "login" ? "auth-switch__button is-active" : "auth-switch__button"}
            onClick={() => setMode("login")}
          >
            Login
          </button>
          <button
            type="button"
            className={mode === "signup" ? "auth-switch__button is-active" : "auth-switch__button"}
            onClick={() => setMode("signup")}
          >
            Signup
          </button>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          {mode === "signup" ? (
            <label className="field">
              <span>Name</span>
              <input
                value={activeForm.name}
                onChange={(event) => updateField("name", event.target.value)}
                placeholder="Aarav Sharma"
                required
              />
            </label>
          ) : null}

          <label className="field">
            <span>Email</span>
            <input
              type="email"
              value={activeForm.email}
              onChange={(event) => updateField("email", event.target.value)}
              placeholder="lab@crystal.ai"
              required
            />
          </label>

          <label className="field">
            <span>Password</span>
            <input
              type="password"
              value={activeForm.password}
              onChange={(event) => updateField("password", event.target.value)}
              placeholder="Minimum 8 characters"
              required
              minLength={8}
            />
          </label>

          {error ? <p className="form-error">{error}</p> : null}

          <button type="submit" className="primary-button" disabled={busy}>
            {busy ? "Authenticating..." : mode === "login" ? "Enter Dashboard" : "Create Account"}
          </button>
        </form>

          <p className="auth-terms">
            By signing up, you agree to our <a href="/terms" target="_blank">Terms of Service</a> and acknowledge our <a href="/privacy" target="_blank">Privacy Policy</a>.
          </p>
      </section>
    </div>
  );
}
