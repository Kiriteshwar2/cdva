import AuthLanding from "../components/AuthLanding";

export default function Login() {
  async function handleLogin(formData) {
    console.log("LOGIN:", formData);

    // TODO:
    // connect backend login API here

    window.location.href = "/dashboard";
  }

  async function handleSignup(formData) {
    console.log("SIGNUP:", formData);

    // TODO:
    // connect backend signup API here

    window.location.href = "/dashboard";
  }

  return (
    <AuthLanding
      onLogin={handleLogin}
      onSignup={handleSignup}
      busy={false}
      error=""
    />
  );
}