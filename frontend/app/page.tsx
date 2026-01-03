"use client";

import { useMemo, useState } from "react";

type Deal = {
  property_id: number;
  deal_type: string;
  city: string;
  state: string;
  beds: number;
  baths: number;
  sqft: number;
  purchase_price: number;
  arv: number;
  entry_fee: number;
  estimated_monthly_payment: number;
  rerank_score: number;
  retrieval_sim: number;
};

type ChatResponse = {
  reply: string;
  deals: Deal[];
  needs_clarification: boolean;
  missing_fields?: string[];
};

type Msg = { role: "user" | "assistant"; content: string };

function money(n: number) {
  try {
    return new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(n);
  } catch {
    return `$${Math.round(n).toLocaleString()}`;
  }
}

export default function Page() {
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const [messages, setMessages] = useState<Msg[]>([
    {
      role: "assistant",
      content:
        "Hi! I’m NexKey DealMatch. Tell me your buy box and I’ll recommend the best deals.\n\nExample: “3 bed in AZ under 350k, entry under 20k, payment under 2500”.",
    },
  ]);

  const [input, setInput] = useState("");
  const [topK, setTopK] = useState(5);
  const [topN, setTopN] = useState(50);
  const [loading, setLoading] = useState(false);
  const [deals, setDeals] = useState<Deal[]>([]);
  const [hint, setHint] = useState<string | null>(null);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  async function send() {
    if (!canSend) return;

    const userText = input.trim();
    setInput("");
    setHint(null);
    setDeals([]);

    setMessages((m) => [...m, { role: "user", content: userText }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText, top_k: topK, top_n: topN }),
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      const data = (await res.json()) as ChatResponse;

      let assistantText = data.reply;
      if (data.needs_clarification && data.missing_fields?.length) {
        assistantText += `\n\nMissing: ${data.missing_fields.join(", ")}`;
        setHint("Try adding the missing fields, like: “3 bed in AZ under 350k”");
      }

      setMessages((m) => [...m, { role: "assistant", content: assistantText }]);
      setDeals(data.deals || []);
    } catch (e: any) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `Sorry — I couldn’t reach the API. Make sure backend is running on ${API_URL}.` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 to-slate-900 text-slate-100">
      <div className="mx-auto max-w-6xl px-4 py-8">
        {/* Header */}
        <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight">NexKey DealMatch</h1>
            <p className="text-slate-300">
              Deep-learning retrieval + reranking chatbot for wholesale real estate deal matching.
            </p>
          </div>

          <div className="flex gap-3 rounded-2xl bg-slate-900/60 p-3 ring-1 ring-white/10">
            <div className="flex flex-col">
              <label className="text-xs text-slate-300">Top K</label>
              <input
                className="w-20 rounded-lg bg-slate-950 px-3 py-2 text-sm ring-1 ring-white/10 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(Math.max(1, Math.min(20, Number(e.target.value))))}
              />
            </div>

            <div className="flex flex-col">
              <label className="text-xs text-slate-300">Retrieve N</label>
              <input
                className="w-24 rounded-lg bg-slate-950 px-3 py-2 text-sm ring-1 ring-white/10 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                type="number"
                min={10}
                max={500}
                value={topN}
                onChange={(e) => setTopN(Math.max(10, Math.min(500, Number(e.target.value))))}
              />
            </div>
          </div>
        </div>

        {/* Main */}
        <div className="mt-8 grid gap-6 lg:grid-cols-5">
          {/* Chat */}
          <div className="lg:col-span-3">
            <div className="rounded-3xl bg-slate-900/60 ring-1 ring-white/10">
              <div className="border-b border-white/10 px-5 py-4">
                <h2 className="text-lg font-medium">Chat</h2>
                <p className="text-sm text-slate-300">Describe your buy box and I’ll recommend deals.</p>
              </div>

              <div className="h-[520px] overflow-y-auto px-5 py-4">
                <div className="flex flex-col gap-4">
                  {messages.map((m, idx) => (
                    <div
                      key={idx}
                      className={`max-w-[90%] rounded-2xl px-4 py-3 text-sm leading-relaxed ring-1 ring-white/10 ${
                        m.role === "user"
                          ? "ml-auto bg-indigo-600/30"
                          : "mr-auto bg-slate-950/40"
                      }`}
                    >
                      <pre className="whitespace-pre-wrap font-sans">{m.content}</pre>
                    </div>
                  ))}

                  {loading && (
                    <div className="mr-auto max-w-[90%] rounded-2xl bg-slate-950/40 px-4 py-3 text-sm ring-1 ring-white/10">
                      Thinking…
                    </div>
                  )}
                </div>
              </div>

              <div className="border-t border-white/10 p-4">
                <div className="flex gap-3">
                  <input
                    className="flex-1 rounded-2xl bg-slate-950 px-4 py-3 text-sm ring-1 ring-white/10 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    placeholder='Try: "3 bed in AZ under 350k, entry under 20k, payment under 2500"'
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") send();
                    }}
                  />
                  <button
                    className={`rounded-2xl px-5 py-3 text-sm font-medium ring-1 ring-white/10 transition ${
                      canSend ? "bg-indigo-600 hover:bg-indigo-500" : "bg-slate-800 opacity-60"
                    }`}
                    onClick={send}
                    disabled={!canSend}
                  >
                    Send
                  </button>
                </div>

                {hint && <p className="mt-2 text-xs text-slate-300">{hint}</p>}
              </div>
            </div>

            <div className="mt-3 text-xs text-slate-400">
              Tip: If you’re vague, the bot will ask follow-up questions. That’s intentional chatbot behavior.
            </div>
          </div>

          {/* Deals panel */}
          <div className="lg:col-span-2">
            <div className="rounded-3xl bg-slate-900/60 ring-1 ring-white/10">
              <div className="border-b border-white/10 px-5 py-4">
                <h2 className="text-lg font-medium">Top Deals</h2>
                <p className="text-sm text-slate-300">Returned by deep-learning reranker.</p>
              </div>

              <div className="p-5">
                {deals.length === 0 ? (
                  <div className="rounded-2xl bg-slate-950/40 p-4 text-sm text-slate-300 ring-1 ring-white/10">
                    When you send a message, I’ll show the top deals here.
                  </div>
                ) : (
                  <div className="flex flex-col gap-4">
                    {deals.map((d) => (
                      <div key={d.property_id} className="rounded-2xl bg-slate-950/40 p-4 ring-1 ring-white/10">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="text-sm font-medium">
                              {d.deal_type} • {d.city}, {d.state}
                            </div>
                            <div className="mt-1 text-xs text-slate-300">
                              {d.beds}bd / {d.baths}ba • {Math.round(d.sqft).toLocaleString()} sqft
                            </div>
                          </div>
                          <div className="rounded-xl bg-indigo-600/20 px-2 py-1 text-xs ring-1 ring-white/10">
                            score {d.rerank_score.toFixed(2)}
                          </div>
                        </div>

                        <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-200">
                          <div className="rounded-xl bg-slate-900/60 p-2 ring-1 ring-white/10">
                            Buy: <span className="font-medium">{money(d.purchase_price)}</span>
                          </div>
                          <div className="rounded-xl bg-slate-900/60 p-2 ring-1 ring-white/10">
                            ARV: <span className="font-medium">{money(d.arv)}</span>
                          </div>
                          <div className="rounded-xl bg-slate-900/60 p-2 ring-1 ring-white/10">
                            Entry: <span className="font-medium">{money(d.entry_fee)}</span>
                          </div>
                          <div className="rounded-xl bg-slate-900/60 p-2 ring-1 ring-white/10">
                            Pay: <span className="font-medium">{money(d.estimated_monthly_payment)}</span>
                          </div>
                        </div>

                        <div className="mt-3 text-[11px] text-slate-400">
                          retrieval sim: {d.retrieval_sim.toFixed(3)} • property_id: {d.property_id}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="mt-3 rounded-3xl bg-slate-900/60 p-5 ring-1 ring-white/10">
              <h3 className="text-sm font-medium">Suggested prompts</h3>
              <ul className="mt-2 space-y-2 text-sm text-slate-300">
                <li>• 3 bed in AZ under 350k, entry under 20k, payment under 2500</li>
                <li>• Phoenix AZ 4 bed under 450k, ARV 550k+, entry under 25k</li>
                <li>• Subto deal in Arizona, 3 bed minimum, under 400k</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-10 text-xs text-slate-500">
          Built with: Dual Encoder retrieval + Cross Encoder reranking • FastAPI • Next.js
        </div>
      </div>
    </div>
  );
}
