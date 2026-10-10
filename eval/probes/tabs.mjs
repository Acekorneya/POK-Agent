// Prints every page tab of a Chrome started with --remote-debugging-port=<port>
// as JSON: title, URL, the visible text (start), and each form control's
// name, type, value, and checked state. Used by eval/run.py to score form runs.
//
// node eval/probes/tabs.mjs 9333

const port = process.argv[2];
const tabs = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();

const expression = `JSON.stringify({
  title: document.title,
  url: location.href,
  text: document.body.innerText.slice(0, 400),
  fields: [...document.querySelectorAll('input, textarea, select')].map(el => ({
    name: el.name || el.getAttribute('aria-label') || el.id,
    type: el.type,
    value: el.value,
    checked: el.checked === true,
  })),
})`;

const pages = [];
for (const tab of tabs.filter(tab => tab.type === 'page')) {
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = reject;
  });
  const reply = await new Promise(resolve => {
    ws.onmessage = event => {
      const message = JSON.parse(event.data);
      if (message.id === 1) resolve(message);
    };
    ws.send(JSON.stringify({ id: 1, method: 'Runtime.evaluate', params: { expression, returnByValue: true } }));
  });
  ws.close();
  const value = reply.result?.result?.value;
  if (value) pages.push({ tab_id: tab.id, ...JSON.parse(value) });
}
console.log(JSON.stringify(pages));
