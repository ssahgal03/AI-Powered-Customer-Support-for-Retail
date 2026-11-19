# Product and policy documents

Use mock content only. Add UTF-8 Markdown or plain-text documents. Name product manuals `<product_id>-manual.md` or `<product_id>-manual.txt`, using IDs from `data/mock_orders.json`. Name store-wide policy files with the `-policy.md` or `-policy.txt` suffix. The filename is used in citations.

The first product question in an online process embeds the corpus using the application's existing OpenAI configuration. The index is stored in memory and rebuilt after corpus changes or a process restart. Subsequent searches embed only the question. Product-scoped searches include that product's manual and all store-wide policies. With no product ID, all indexed documents are eligible. Offline mode and embedding service failures produce an empty result so the product information branch can use its existing fallback.
