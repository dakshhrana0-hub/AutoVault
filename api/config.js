export default function handler(req, res) {
  res.setHeader('Content-Type', 'application/javascript');
  
  const supabaseUrl = process.env.SUPABASE_URL || '';
  const supabaseAnon = process.env.SUPABASE_ANON || '';

  // Return exactly the JS script that the frontend expects
  const scriptContent = `
const SUPABASE_URL = "${supabaseUrl}";
const SUPABASE_ANON = "${supabaseAnon}";
// DO NOT ADD GROQ_API_KEY HERE. IT REMAINS SECURELY ON THE SERVER.
`;

  res.status(200).send(scriptContent);
}
