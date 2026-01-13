const expressionInput = document.getElementById("expression");
const variableInput = document.getElementById("variable");
const orderInput = document.getElementById("order");
const lowerInput = document.getElementById("lower");
const upperInput = document.getElementById("upper");
const valueInput = document.getElementById("value");
const variablesInput = document.getElementById("variables");
const historyPane = document.getElementById("history");
const resultText = document.getElementById("result-text");
const approxLine = document.getElementById("approx-line");
const approxValue = document.getElementById("approx-value");
const activeModeLabel = document.getElementById("active-mode");
const touchExpression = document.getElementById("touch-expression");
const copyResultButton = document.getElementById("copy-result");
const copyExpressionButton = document.getElementById("copy-expression");
const executeButton = document.getElementById("execute");

const MODE_LABELS = {
    derivative: "DIFF",
    evaluate: "CAS",
    integral: "INTEGRAL",
    limit: "LIMIT",
    simplify: "ALG",
    solve_equation: "SOLVE",
    solve_ode: "ODE",
    solve_system: "SYSTEM",
    taylor_series: "SERIES",
};

let currentMode = "evaluate";
let lastResultToken = null;

function setMode(mode) {
    currentMode = mode;
    activeModeLabel.textContent = MODE_LABELS[mode] ?? mode.toUpperCase();
    document.querySelectorAll(".mode-button").forEach((button) => {
        button.classList.toggle("active", button.dataset.mode === mode);
    });
    document.querySelectorAll(".soft-key").forEach((button) => {
        button.classList.toggle("active", button.dataset.mode === mode);
    });
}

function appendToken(token) {
    const cursor = expressionInput.selectionStart ?? expressionInput.value.length;
    const before = expressionInput.value.slice(0, cursor);
    const after = expressionInput.value.slice(cursor);
    expressionInput.value = `${before}${token}${after}`;
    const newPosition = cursor + token.length;
    expressionInput.setSelectionRange(newPosition, newPosition);
    expressionInput.focus();
    renderTouchExpression();
}

function deleteToken() {
    const cursor = expressionInput.selectionStart ?? expressionInput.value.length;
    if (cursor === 0) return;
    const before = expressionInput.value.slice(0, cursor - 1);
    const after = expressionInput.value.slice(cursor);
    expressionInput.value = `${before}${after}`;
    expressionInput.setSelectionRange(cursor - 1, cursor - 1);
    expressionInput.focus();
    renderTouchExpression();
}

async function executeCalculation() {
    const payload = buildPayload();
    resultText.textContent = "Working…";
    approxLine.hidden = true;

    // Loading state
    const originalText = executeButton.textContent;
    executeButton.textContent = "Processing...";
    executeButton.disabled = true;
    executeButton.setAttribute("aria-busy", "true");

    try {
        const response = await fetch("/api/calculate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Calculation failed");
        }

        renderResult(data);
        pushHistory(payload.expression, data.result);
    } catch (error) {
        resultText.textContent = error.message;
    } finally {
        executeButton.textContent = originalText;
        executeButton.disabled = false;
        executeButton.removeAttribute("aria-busy");
        expressionInput.focus();
    }
}

function buildPayload() {
    const basePayload = {
        operation: currentMode,
        expression: expressionInput.value.trim(),
        variable: variableInput.value.trim() || undefined,
    };

    const substitutions = parseVariableAssignments(variablesInput.value.trim());
    if (lastResultToken) {
        substitutions.ans = lastResultToken;
    }
    if (Object.keys(substitutions).length > 0) {
        basePayload.variables = substitutions;
    }

    if (orderInput.value) {
        basePayload.order = Number(orderInput.value);
    }

    if (lowerInput.value) {
        basePayload.lower = lowerInput.value.trim();
    }

    if (upperInput.value) {
        basePayload.upper = upperInput.value.trim();
    }

    if (valueInput.value) {
        basePayload.value = valueInput.value.trim();
        basePayload.around = valueInput.value.trim();
    }

    if (currentMode === "limit") {
        basePayload.direction = "two-sided";
    }

    if (currentMode === "solve_ode") {
        basePayload.function = variableInput.value.trim() || "f";
    }

    return basePayload;
}

function parseVariableAssignments(raw) {
    if (!raw) return {};
    return raw.split(",").reduce((accumulator, part) => {
        const [name, value] = part.split("=").map((piece) => piece.trim());
        if (name && value) {
            accumulator[name] = value;
        }
        return accumulator;
    }, {});
}

function renderResult(data) {
    const pretty = data.pretty ? `${data.pretty}` : `${data.result}`;
    resultText.textContent = pretty;

    if (typeof data.result === "string" || typeof data.result === "number") {
        lastResultToken = `${data.result}`;
    } else {
        lastResultToken = null;
    }

    if (typeof data.approximation === "number") {
        approxValue.textContent = data.approximation;
        approxLine.hidden = false;
    } else {
        approxLine.hidden = true;
    }
}

async function copyToClipboard(text, label) {
    if (!text) return;
    try {
        await navigator.clipboard.writeText(text);
        resultText.dataset.copied = label;
        setTimeout(() => delete resultText.dataset.copied, 1200);
    } catch (error) {
        resultText.dataset.copied = "Clipboard unavailable";
        setTimeout(() => delete resultText.dataset.copied, 1200);
    }
}

function pushHistory(expression, result) {
    const commandLine = document.createElement("div");
    commandLine.className = "history-line command";
    commandLine.textContent = `▶ ${expression}`;
    commandLine.dataset.expression = expression;

    const answerLine = document.createElement("div");
    answerLine.className = "history-line answer";
    answerLine.textContent = `= ${result}`;
    answerLine.dataset.result = `${result}`;

    historyPane.prepend(answerLine);
    historyPane.prepend(commandLine);

    const lines = historyPane.querySelectorAll(".history-line");
    if (lines.length > 10) {
        Array.from(lines)
            .slice(10)
            .forEach((line) => line.remove());
    }
}

function registerEvents() {
    document.querySelectorAll(".soft-key").forEach((button) => {
        button.addEventListener("click", () => setMode(button.dataset.mode));
    });

    document.querySelectorAll(".mode-button").forEach((button) => {
        button.addEventListener("click", () => setMode(button.dataset.mode));
    });

    document.querySelectorAll("[data-token]").forEach((key) => {
        key.addEventListener("click", () => appendToken(key.dataset.token));
    });

    document.querySelectorAll("[data-action]").forEach((key) => {
        key.addEventListener("click", () => handleAction(key.dataset.action));
    });

    document.getElementById("delete").addEventListener("click", deleteToken);

    const clearButton = document.getElementById("clear");
    let clearConfirmTimeout = null;

    clearButton.addEventListener("click", () => {
        if (clearButton.dataset.confirming) {
            expressionInput.value = "";
            variableInput.value = "";
            orderInput.value = "";
            lowerInput.value = "";
            upperInput.value = "";
            valueInput.value = "";
            variablesInput.value = "";
            resultText.textContent = "Ready.";
            approxLine.hidden = true;
            renderTouchExpression();

            clearButton.textContent = "CLEAR";
            delete clearButton.dataset.confirming;
            clearTimeout(clearConfirmTimeout);
        } else {
            clearButton.textContent = "CONFIRM?";
            clearButton.dataset.confirming = "true";

            clearConfirmTimeout = setTimeout(() => {
                clearButton.textContent = "CLEAR";
                delete clearButton.dataset.confirming;
            }, 3000);
        }
    });

    copyResultButton?.addEventListener("click", () =>
        copyToClipboard(resultText.textContent, "Result copied")
    );
    copyExpressionButton?.addEventListener("click", () =>
        copyToClipboard(expressionInput.value, "Input copied")
    );

    document.getElementById("execute").addEventListener("click", executeCalculation);

    expressionInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            executeCalculation();
        }
    });

    expressionInput.addEventListener("input", renderTouchExpression);
    historyPane.addEventListener("click", handleHistoryRecall);
    resultText.addEventListener("click", () => {
        if (!resultText.textContent) return;
        expressionInput.value = resultText.textContent;
        renderTouchExpression();
        expressionInput.focus();
    });

    document.querySelectorAll("[data-touch-action]").forEach((button) => {
        button.addEventListener("click", () => handleTouchAction(button.dataset.touchAction));
    });

    if (touchExpression) {
        touchExpression.addEventListener("pointerdown", placeCursorFromTouch);
        touchExpression.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                expressionInput.focus();
            }
        });
    }
}

function bootDisplay() {
    historyPane.innerHTML = "";
    const bootLine = document.createElement("div");
    bootLine.className = "history-line status";
    bootLine.textContent = "Aurora CAS ready";
    historyPane.append(bootLine);
    resultText.textContent = "0 ▸";
}

function handleAction(action) {
    const cursor = expressionInput.selectionStart ?? 0;
    if (action === "cursor-left") {
        const newPosition = Math.max(0, cursor - 1);
        expressionInput.setSelectionRange(newPosition, newPosition);
    }
    if (action === "cursor-right") {
        const newPosition = Math.min(expressionInput.value.length, cursor + 1);
        expressionInput.setSelectionRange(newPosition, newPosition);
    }
    if (action === "cursor-start") {
        expressionInput.setSelectionRange(0, 0);
    }
    if (action === "cursor-end") {
        const end = expressionInput.value.length;
        expressionInput.setSelectionRange(end, end);
    }
    expressionInput.focus();
    renderTouchExpression();
}

function renderTouchExpression() {
    if (!touchExpression) return;
    touchExpression.innerHTML = "";
    const text = expressionInput.value;
    if (!text) {
        touchExpression.textContent = "Tap anywhere on the screen to start editing";
        return;
    }

    Array.from(text).forEach((character, index) => {
        const span = document.createElement("span");
        span.dataset.index = `${index}`;
        span.className = "touch-char";
        span.textContent = character;
        touchExpression.append(span);
    });
}

function placeCursorFromTouch(event) {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
        return;
    }
    const index = target.dataset.index ? Number(target.dataset.index) + 1 : null;
    const position = Number.isFinite(index)
        ? Math.min(Math.max(index, 0), expressionInput.value.length)
        : expressionInput.value.length;
    expressionInput.setSelectionRange(position, position);
    expressionInput.focus();
}

function handleHistoryRecall(event) {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const line = target.closest(".history-line");
    if (!(line instanceof HTMLElement)) return;
    const expression = line.dataset.expression || line.dataset.result;
    if (!expression) return;
    if (event.metaKey || event.ctrlKey) {
        copyToClipboard(expression.replace(/^▶\s*/, "").replace(/^=\s*/, ""), "Copied");
        return;
    }
    expressionInput.value = expression.replace(/^▶\s*/, "").replace(/^=\s*/, "");
    renderTouchExpression();
    expressionInput.focus();
}

function handleTouchAction(action) {
    if (action === "backspace") {
        deleteToken();
        return;
    }
    if (action === "insert-ans" && lastResultToken) {
        appendToken(lastResultToken);
        return;
    }
    handleAction(action ?? "");
}

if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
        navigator.serviceWorker.register("/service-worker.js").catch(() => {
            /* Service worker registration failed silently to keep offline optional */
        });
    });
}

registerEvents();
bootDisplay();
setMode(currentMode);
renderTouchExpression();
