-- Lesson content.
--
-- Structure each step follows:
--   * one idea per step, nothing more
--   * the concrete problem before the abstract name for it
--   * plain words; a term is used only after it has been earned
--   * the last step lands on the hardest question in the bank for the concept,
--     so practice picks up exactly where teaching left off
--
-- Re-runnable: ON CONFLICT updates the text, so revising a step is just
-- editing this file and applying it again:
--     docker exec -i flowtutor-db psql -U flowtutor -d flowtutor < backend/lessons.sql
--
-- A concept with no rows here has no teaching mode and goes straight to
-- practice, so content can be authored one concept at a time.

INSERT INTO lessons (concept_id, step, title, body) VALUES

(1, 1, 'The problem: answering the same question a thousand times',
'Picture a profile page. Every visitor triggers the same database query: get user 42. A thousand visitors means a thousand identical queries returning a thousand identical answers, each costing the database perhaps 20 milliseconds of work.

Nothing about that answer changed between the first visitor and the thousandth. The database is redoing the same work over and over, and every visitor waits for it.

That is the entire problem caching solves. And notice what makes it solvable: the answer is the same every time. If every visitor asked a different question, there would be nothing to reuse.'),

(1, 2, 'The fix: keep the answer where you can reach it',
'The first time someone asks for user 42, do the work -- query the database, get the answer. Then, before handing it back, keep a copy somewhere fast.

When the next visitor asks the same question, you already have the answer. Hand back the copy and never touch the database.

That copy is a cache. It is not a new kind of database and it holds no truth of its own. It is a shortcut: a place to keep an answer you expect to be asked for again. The database remains the only thing that actually knows.'),

(1, 3, 'Why the shortcut is so much faster',
'Speed here is about distance.

  * Reading from memory on the same machine: well under a microsecond.
  * Reading from a cache server over the local network: a few hundred microseconds.
  * Asking a database that must hit disk, plan a query and send results back: single-digit to tens of milliseconds.

Three different orders of magnitude. So a cache does not make the database faster -- it makes most requests never reach the database at all.

The fraction of requests answered from the cache is called the hit rate. It is the one number that tells you whether a cache is earning its keep.'),

(1, 4, 'What you gave up: the copy can go stale',
'Someone updates the display name of user 42. The database now holds the new name. Your cache still holds the old one, because nobody told it anything had changed.

Every visitor served from the cache sees the old name, and will keep seeing it until something removes that copy.

This is not a bug to be fixed. It is the deal you made: a cache trades freshness for speed. Every real caching decision is a decision about how much staleness you can tolerate, and for how long. If the answer is none, ever, then that data cannot be cached.'),

(1, 5, 'Two ways to deal with staleness',
'The first is expiry, usually called TTL -- time to live. Stamp each cached copy with a lifetime, say sixty seconds, and discard it when the time is up. Simple, needs cooperation from nobody, and it bounds the damage: data can be wrong, but never for longer than the lifetime you chose.

The second is invalidation. When the write happens, actively delete the cached copy. Fresher, but it requires the write path to know which cached entries the change affects -- and missing one is where stale-data bugs are born.

Most real systems use both: invalidate what they can track, and let TTL clean up what they missed.'),

(1, 6, 'Who actually fills the cache?',
'The usual answer is: your own application code, and only when it has to.

Look in the cache. If the answer is there -- a hit -- return it. If it is not -- a miss -- query the database, put the answer into the cache, then return it.

That is the whole pattern. It is called cache-aside, or lazy loading.

Notice how passive the cache is. It never reaches out to the database, and it does not know the database exists. It is a box you put things into and take things out of. Everything clever happens in your code.'),

(1, 7, 'What breaks at scale: the stampede',
'A popular key expires. In that same instant a thousand requests arrive. All of them look in the cache, all of them miss, and all of them go to the database at once -- delivering the exact load the cache existed to prevent, compressed into one spike.

This is a cache stampede, also called a thundering herd.

Two standard fixes, usually applied together:

  * Let only one request perform the refill while the others wait for its result, so a thousand misses cause one query.
  * Add randomness to expiry times, so keys written together do not all expire together.

This is the point where caching stops being a trick and starts being engineering.')

ON CONFLICT (concept_id, step) DO UPDATE
  SET title = EXCLUDED.title, body = EXCLUDED.body;
