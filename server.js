import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";
import fs from "fs";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatGroq } from "@langchain/groq";
import { tool } from "@langchain/core/tools";          // FIX 1: correct import
import { createReactAgent } from "@langchain/langgraph/prebuilt"; // FIX 1: correct agent import
import { HumanMessage } from "@langchain/core/messages";
import { z } from "zod";

// FIX 3: dotenv before everything else
dotenv.config();

const menuData = JSON.parse(fs.readFileSync("./menu.json", "utf8"));

const app = express();
const port = 3000;

app.use(cors());
app.use(express.json());

const __dirname = path.resolve();

/* -----------------------------
   Gemini Model
----------------------------- */

const model = new ChatGroq({
  model: "qwen/qwen3-32b",
  temperature: 0.7,
  apiKey: process.env.GROQ_API_KEY,
});

/* -----------------------------
   In-Memory Storage
----------------------------- */

const bag = [];
const orders = [];

/* -----------------------------
   Helper: format item for LLM
----------------------------- */

function formatItem(item) {
  let priceText = "";
  if (item.price.full)  priceText += `Full ₹${item.price.full}`;
  if (item.price.half)  priceText += ` | Half ₹${item.price.half}`;
  return `• ${item.name} (${item.category}, ${item.veg_nonveg}, Spice: ${item.spice_level}) - ${priceText}`;
}

/* -----------------------------
   Tools
----------------------------- */

const searchMenuTool = tool(
  async ({ query }) => {
    const q = query.toLowerCase();
    const results = menuData.filter(item =>
      item.name.toLowerCase().includes(q) ||
      item.category.toLowerCase().includes(q) ||
      item.main_ingredient.toLowerCase().includes(q) ||
      item.spice_level.toLowerCase().includes(q) ||
      item.veg_nonveg.toLowerCase().includes(q)
    );
    if (!results.length) return "No matching dishes found.";
    const sliced = results.slice(0, 10);
    let response = sliced.map(formatItem).join("\n");
    if (results.length > 10) {
      response += `\n\n(Showing first 10 of ${results.length} matches. Please narrow down your search query for more specific results.)`;
    }
    return response;
  },
  {
    name: "search_menu",
    description: "Search restaurant menu by name, ingredient, category, spice level or veg/nonveg",
    schema: z.object({
      query: z.string()
    })
  }
);

const filterMenuTool = tool(
  // FIX 2: maxPrice added to destructured args
  async ({ category, veg_nonveg, spice_level, main_ingredient, maxPrice, avoid_ingredients }) => {
    console.log("FILTER TOOL CALLED", { category, veg_nonveg, spice_level, main_ingredient, maxPrice, avoid_ingredients });

    const results = menuData.filter(item => {
      if (category && item.category.toLowerCase() !== category.toLowerCase()) return false;
      if (veg_nonveg && item.veg_nonveg.toLowerCase() !== veg_nonveg.toLowerCase()) return false;
      if (spice_level && item.spice_level.toLowerCase() !== spice_level.toLowerCase()) return false;
      if (main_ingredient && item.main_ingredient.toLowerCase() !== main_ingredient.toLowerCase()) return false;

      if (maxPrice) {
        const lowestPrice = item.price.half || item.price.full;
        if (lowestPrice > maxPrice) return false;
      }

      if (avoid_ingredients && avoid_ingredients.length > 0) {
        const hasAllergen = item.ingredients.some(ing =>
          avoid_ingredients.map(i => i.toLowerCase()).includes(ing.toLowerCase())
        );
        if (hasAllergen) return false;
      }

      return true;
    });

    if (!results.length) return "No dishes match those preferences.";
    const sliced = results.slice(0, 10);
    let response = sliced.map(formatItem).join("\n");
    if (results.length > 10) {
      response += `\n\n(Showing first 10 of ${results.length} matching preferences. Please refine your filters to narrow down the list.)`;
    }
    return response;
  },
  {
    name: "filter_menu",
    description: "Filter menu by category, veg/nonveg, spice level, main ingredient, max price and ingredients to avoid",
    schema: z.object({
      category: z.string().optional(),
      veg_nonveg: z.string().optional(),
      spice_level: z.string().optional(),
      main_ingredient: z.string().optional(),
      maxPrice: z.number().optional(),
      avoid_ingredients: z.array(z.string()).optional()
    })
  }
);

const addToBagTool = tool(
  async ({ itemName }) => {
    const item = menuData.find(dish => dish.name.toLowerCase() === itemName.toLowerCase());
    if (!item) return `Item "${itemName}" not found in menu.`;
    bag.push(item);
    return `${item.name} added to bag. Bag now has ${bag.length} item(s).`;
  },
  {
    name: "add_to_bag",
    description: "Add a dish to the customer's bag/cart by its exact name",
    schema: z.object({
      itemName: z.string()
    })
  }
);

const viewBagTool = tool(
  async () => {
    if (!bag.length) return "Your bag is empty.";
    const total = bag.reduce((sum, item) => sum + (item.price.full || item.price.half || 0), 0);
    return `Items in bag:\n${bag.map(i => `• ${i.name} - ₹${i.price.full || i.price.half}`).join("\n")}\n\nTotal: ₹${total}`;
  },
  {
    name: "view_cart",
    description: "View all items currently in the customer's bag and the total price",
    schema: z.object({})  // FIX 5: schema required even for no-arg tools
  }
);

const placeOrderTool = tool(
  async () => {
    if (!bag.length) return "Your bag is empty. Add items before placing an order.";

    const token = "T" + Math.floor(1000 + Math.random() * 9000);
    const total = bag.reduce((sum, item) => sum + (item.price.full || item.price.half || 0), 0);

    orders.push({
      token,
      items: [...bag],
      total,
      status: "Pending",
      createdAt: new Date()
    });

    bag.length = 0;

    return `✅ Order placed!\n\nToken: ${token}\nTotal: ₹${total}\nStatus: Pending\n\nSave your token to check order status.`;
  },
  {
    name: "place_order",
    description: "Place the current bag as an order. Returns a token number to track the order.",
    schema: z.object({})  // FIX 5: schema required even for no-arg tools
  }
);

const checkStatusTool = tool(
  async ({ token }) => {
    const order = orders.find(o => o.token.toLowerCase() === token.toLowerCase());
    if (!order) return `❌ No order found with token "${token}". Please check the token and try again.`;
    return `🍽 Order Status\n\nToken: ${order.token}\nItems: ${order.items.map(i => i.name).join(", ")}\nTotal: ₹${order.total}\nStatus: ${order.status}`;
  },
  {
    name: "check_status",
    description: "Check the status of a placed order using its token number",
    schema: z.object({
      token: z.string()
    })
  }
);

/* -----------------------------
   Agent
   FIX 1: createReactAgent (not createAgent)
   FIX 7: full system prompt covering all tools
----------------------------- */

const agent = createReactAgent({
  llm: model,
  tools: [searchMenuTool, filterMenuTool, addToBagTool, viewBagTool, placeOrderTool, checkStatusTool],
  stateModifier: `
You are Handi, a warm and helpful restaurant AI assistant for Handi Restaurant.

Guidelines:
- When recommending dishes, show the name, category, spice level, and price.
- Use the available tools to search/filter the menu, manage the customer's bag, place orders, and check order status.
- When invoking tools, only pass arguments that are explicitly needed. Omit unneeded optional arguments (do not pass them as empty strings or empty arrays).
- When a customer wants to add an item to their order, use the 'add_to_bag' tool for that item.
- Always view the bag contents or confirm with the customer before placing an order.
- Always use the tools to place the order or check status.
- Never make up menu items or prices; always retrieve them using the provided tools.
- Keep your tone warm, polite, and helpful.
`.trim()
});

/* -----------------------------
   Routes
----------------------------- */

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.get("/admin", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "admin.html"));
});

/* -----------------------------
   Bag/Cart Sync APIs
----------------------------- */

app.get("/api/bag", (req, res) => {
  res.json(bag);
});

app.post("/api/bag", (req, res) => {
  const { id } = req.body;
  const item = menuData.find(m => m.id === Number(id));
  if (!item) return res.status(404).json({ error: "Item not found" });
  bag.push(item);
  res.json(bag);
});

app.delete("/api/bag/:id", (req, res) => {
  const id = Number(req.params.id);
  const idx = bag.findIndex(m => m.id === id);
  if (idx >= 0) {
    bag.splice(idx, 1);
  }
  res.json(bag);
});

/* -----------------------------
   Chat Endpoint
   FIX 4: support conversation history
----------------------------- */

app.post("/chat", async (req, res) => {
  try {
    // Frontend should send { messages: [{role, content}, ...] }
    // Falls back to { message: "..." } for backwards compatibility
    const { message, messages } = req.body;

    const history = messages
      ? messages.map(m => new HumanMessage(m.content))  // simplified; extend for multi-role if needed
      : [new HumanMessage(message)];

    const result = await agent.invoke({ messages: history });

    const reply = result.messages[result.messages.length - 1].content;

    res.json({ reply, bag });
  } catch (error) {
    console.error(error);
    res.status(500).json({ reply: "Something went wrong. Please try again." });
  }
});

/* -----------------------------
   Admin APIs
----------------------------- */

app.get("/admin/orders", (req, res) => {
  res.json(orders);
});

app.put("/admin/order/:token", (req, res) => {
  const { token } = req.params;
  const { status } = req.body;

  const order = orders.find(o => o.token === token);
  if (!order) return res.status(404).json({ message: "Order not found" });

  order.status = status;
  res.json(order);
});

/* -----------------------------
   Customer Status API
----------------------------- */

app.get("/order/:token", (req, res) => {
  const order = orders.find(o => o.token === req.params.token);
  if (!order) return res.status(404).json({ message: "Order not found" });
  res.json(order);
});

/* -----------------------------
   Start Server
----------------------------- */

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});