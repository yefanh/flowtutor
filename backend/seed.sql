-- Phase 0 seed content: 5 concepts x 22 questions, spread across difficulty 1..5.
--
-- Difficulty is spread deliberately: Phase 1's adaptive engine needs several
-- rungs per concept, otherwise there is nothing to step up or down to.
--
-- This file is IDEMPOTENT -- every statement ends in ON CONFLICT DO NOTHING,
-- so re-running it after a schema migration reloads content without creating
-- duplicates. Apply it with:
--     docker exec -i flowtutor-db psql -U flowtutor -d flowtutor < backend/seed.sql

INSERT INTO concepts (id, name, description) VALUES
  (1, 'Caching',
      'Storing computed or fetched results closer to the consumer to cut latency and load.'),
  (2, 'Message Queues',
      'Asynchronous, decoupled communication between services via durable queues and topics.'),
  (3, 'Consistency Models',
      'What guarantees a distributed store gives about the order and visibility of reads and writes.'),
  (4, 'Distributed Systems Fundamentals',
      'Scaling, partitioning, coordination and failure handling across many machines.'),
  (5, 'RAG and Agents',
      'Grounding language models with retrieval, and wrapping them in tool-using control loops.')
ON CONFLICT DO NOTHING;

SELECT setval('concepts_id_seq', (SELECT MAX(id) FROM concepts));


-- ---------------------------------------------------------------- Caching (1)
INSERT INTO questions (concept_id, stem, options, answer, difficulty, explanation) VALUES
(1, 'What is the primary reason to put a cache in front of a database?',
 '["To reduce read latency and load on the database", "To guarantee the data is always correct", "To provide durable long-term storage", "To enforce access control on the data"]',
 0, 1,
 'A cache trades freshness for speed. It serves hot reads quickly and shields the database from load; it is not a source of truth and gives no durability or correctness guarantee.'),

(1, 'In the cache-aside pattern, who is responsible for populating the cache?',
 '["The database, via a trigger", "The application code, on a cache miss", "The cache itself, by prefetching", "A background job, on a fixed schedule"]',
 1, 2,
 'Cache-aside (lazy loading) means the application checks the cache, and on a miss reads the database and writes the value back into the cache itself. The cache stays passive.'),

(1, 'A key expires and thousands of concurrent requests all miss and hit the database at once. What is this called, and what is a standard mitigation?',
 '["Cache pollution; use a larger cache", "Cache stampede; use a lock or single-flight so one request repopulates", "Cache coherence failure; disable the cache", "Cold start; increase the TTL to infinity"]',
 1, 3,
 'This is a cache stampede (or thundering herd). The usual fixes are single-flight or a mutex so only one request recomputes, plus randomized TTL jitter so keys do not all expire together.'),

(1, 'What is the key difference between write-through and write-back caching?',
 '["Write-through writes to cache and store synchronously; write-back defers the store write", "Write-through is faster on writes; write-back is faster on reads", "Write-through works only for reads; write-back only for writes", "There is no difference; the terms are interchangeable"]',
 0, 4,
 'Write-through keeps cache and backing store in sync on every write, so it is safer but slower. Write-back acknowledges the write after the cache only and flushes later, which is faster but can lose data on failure.'),

(1, 'Your cache hit rate is high but users still report stale data after updates. Which change most directly addresses this?',
 '["Increase the cache size", "Switch from LRU to LFU eviction", "Invalidate or update the cache key on write, rather than waiting for the TTL", "Add a second cache layer in front of the first"]',
 2, 5,
 'Staleness after a write is an invalidation problem, not a capacity or eviction problem. The write path has to invalidate or update the affected keys; TTL alone only bounds how long the staleness lasts.')
ON CONFLICT DO NOTHING;


-- --------------------------------------------------------- Message Queues (2)
INSERT INTO questions (concept_id, stem, options, answer, difficulty, explanation) VALUES
(2, 'What is the main benefit of putting a message queue between two services?',
 '["It makes the overall request faster end to end", "It decouples the producer from the consumer, so they can fail and scale independently", "It removes the need for a database", "It guarantees messages are processed in exactly one order"]',
 1, 1,
 'A queue buys decoupling: the producer does not need the consumer to be up, fast, or scaled the same way. End-to-end latency usually gets worse, not better.'),

(2, 'A queue guarantees at-least-once delivery. What must consumers therefore be?',
 '["Stateless", "Idempotent", "Single-threaded", "Ordered"]',
 1, 2,
 'At-least-once means a message can be redelivered after a failure or timeout. Consumers must be idempotent so that processing the same message twice has the same effect as processing it once.'),

(2, 'In a partitioned log like Kafka, what does ordering actually guarantee?',
 '["Global ordering across the whole topic", "Ordering within a single partition only", "Ordering by message timestamp across all partitions", "No ordering guarantee of any kind"]',
 1, 3,
 'Order is preserved per partition, not per topic. That is why the partition key matters: messages that must stay ordered relative to each other need to hash to the same partition.'),

(2, 'Consumers are falling behind and the queue depth is growing without bound. Which response addresses the root cause rather than the symptom?',
 '["Increase the queue retention period", "Drop the oldest messages", "Scale out consumers or reduce per-message work so throughput exceeds arrival rate", "Increase the producer batch size"]',
 2, 4,
 'A growing backlog means arrival rate exceeds processing rate. Only raising consumer throughput (or lowering arrival rate) fixes it; retention and dropping just change how the overflow is absorbed.')
ON CONFLICT DO NOTHING;


-- ------------------------------------------------------ Consistency Models (3)
INSERT INTO questions (concept_id, stem, options, answer, difficulty, explanation) VALUES
(3, 'The CAP theorem says that during a network partition, a distributed system must choose between which two properties?',
 '["Consistency and availability", "Consistency and performance", "Availability and durability", "Partition tolerance and latency"]',
 0, 1,
 'Partition tolerance is not optional for a real distributed system, so CAP is really a choice made during a partition: stay consistent and refuse some requests, or stay available and serve possibly stale data.'),

(3, 'What does eventual consistency actually promise?',
 '["Reads always return the most recent write", "If writes stop, all replicas eventually converge to the same value", "Writes are never lost under any failure", "Every read is served by the leader replica"]',
 1, 2,
 'Eventual consistency is a convergence promise with no bound on when. It says nothing about what an individual read sees in the meantime.'),

(3, 'A user updates their profile, immediately reloads, and sees the old value. Which consistency guarantee would have prevented this specific complaint?',
 '["Read-your-writes consistency", "Monotonic reads", "Linearizability across all clients", "Strict serializability of transactions"]',
 0, 3,
 'This is exactly the read-your-writes (session) guarantee: a client must see its own prior writes. It is much cheaper to provide than full linearizability, which would also constrain what every other client sees.'),

(3, 'What is the difference between linearizability and serializability?',
 '["They are two names for the same guarantee", "Linearizability is about the real-time order of single-object operations; serializability is about transactions being equivalent to some serial order", "Linearizability applies to transactions; serializability applies to single objects", "Serializability is strictly weaker in every dimension"]',
 1, 5,
 'Linearizability is a recency guarantee on individual objects that respects real time. Serializability is an isolation guarantee on multi-object transactions and does not by itself constrain real-time order. Strict serializability is the combination of both.')
ON CONFLICT DO NOTHING;


-- ------------------------------------ Distributed Systems Fundamentals (4)
INSERT INTO questions (concept_id, stem, options, answer, difficulty, explanation) VALUES
(4, 'What distinguishes horizontal scaling from vertical scaling?',
 '["Horizontal adds more machines; vertical adds more resources to one machine", "Horizontal adds more resources to one machine; vertical adds more machines", "Horizontal applies only to databases; vertical only to web servers", "They describe the same thing at different layers"]',
 0, 1,
 'Horizontal scaling means more nodes, which requires the workload to be partitionable. Vertical scaling means a bigger node, which is simpler but has a hard ceiling and a single failure domain.'),

(4, 'Why is consistent hashing preferred over plain modulo hashing for distributing keys across nodes?',
 '["It distributes keys more evenly under all conditions", "It only remaps a small fraction of keys when a node is added or removed", "It removes the need for replication", "It makes lookups O(1) instead of O(log n)"]',
 1, 2,
 'With modulo hashing, changing the node count remaps nearly every key. Consistent hashing bounds the churn to roughly the share owned by the node that joined or left, which is what makes elastic clusters practical.'),

(4, 'A replicated store uses 5 nodes and requires a quorum for both reads and writes. What is the smallest quorum size that guarantees a read overlaps every committed write?',
 '["2", "3", "4", "5"]',
 1, 3,
 'With N=5, any read and write quorum must satisfy R + W > N. Choosing R = W = 3 gives 6 > 5, so the sets always overlap by at least one node that has the latest write.'),

(4, 'Why should a retry policy use exponential backoff WITH jitter rather than backoff alone?',
 '["Jitter reduces the total number of retries", "Jitter prevents synchronized clients from retrying in lockstep and re-creating the load spike", "Jitter makes each individual retry faster", "Jitter is only needed for single-client systems"]',
 1, 4,
 'Pure exponential backoff still leaves many clients retrying at the same instants after a shared failure. Randomized jitter spreads those retries out, which is what actually stops the retry storm.')
ON CONFLICT DO NOTHING;


-- ------------------------------------------------------- RAG and Agents (5)
INSERT INTO questions (concept_id, stem, options, answer, difficulty, explanation) VALUES
(5, 'What problem does retrieval-augmented generation (RAG) primarily solve?',
 '["It makes the model run faster", "It grounds the model in specific external knowledge it was not trained on, reducing fabrication", "It reduces the number of parameters the model needs", "It replaces the need for prompting"]',
 1, 1,
 'RAG supplies relevant source material at inference time so answers are grounded in real documents instead of the model''s parametric memory. It is about accuracy and freshness, not speed or model size.'),

(5, 'Why is source material split into chunks before being embedded?',
 '["Because embedding models have a limited input length, and smaller units retrieve more precisely", "Because chunks compress better on disk", "Because the database cannot store long text", "Because chunking guarantees the model will not hallucinate"]',
 0, 2,
 'Chunking respects the embedding model input limit and keeps each vector focused on one idea, so a match is actually about the passage you retrieved rather than a whole document averaged together.'),

(5, 'Why combine dense vector search with keyword search (BM25) instead of using dense search alone?',
 '["Keyword search is always more accurate", "Dense search catches paraphrases but can miss exact rare terms; keyword search covers that gap", "Keyword search is required by pgvector", "It halves the storage needed for the index"]',
 1, 3,
 'The two methods fail in different ways. Dense embeddings match meaning but blur exact identifiers, error codes and rare terms; BM25 nails those literal matches. Hybrid retrieval takes the union of their strengths.'),

(5, 'In a retrieval pipeline, what job does a reranker do?',
 '["It re-embeds the query with a larger model", "It reorders an already retrieved candidate set by scoring each document against the query directly", "It removes duplicate documents from the corpus", "It decides how many chunks to store"]',
 1, 4,
 'Retrieval is optimized for recall over a huge corpus and is therefore approximate. A reranker takes the small candidate set and scores each document jointly with the query, which is far more accurate but too slow to run over everything.'),

(5, 'What most fundamentally distinguishes an agent from a single-shot RAG chatbot?',
 '["The agent uses a larger model", "The agent runs a loop where the model itself decides which tool to call next based on prior observations", "The agent retrieves more documents per query", "The agent stores its prompts in a database"]',
 1, 5,
 'An agent is a control loop around the model: plan, act by calling a tool, observe the result, repeat. Control flow is decided by the model at run time, whereas a single-shot RAG pipeline has a fixed sequence of steps hard-coded by the developer.')
ON CONFLICT DO NOTHING;
