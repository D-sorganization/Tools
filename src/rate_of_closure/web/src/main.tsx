import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import "./index.css";
import { applicationWebRuntime } from "./model/webRuntime";

const root = ReactDOM.createRoot(document.getElementById("root")!);
try {
  const runtime = applicationWebRuntime(document);
  root.render(
    <React.StrictMode>
      <App runtime={runtime} />
    </React.StrictMode>,
  );
} catch {
  root.render(
    <main role="alert">
      Rate of Closure cannot start because its runtime descriptor is invalid.
    </main>,
  );
}
