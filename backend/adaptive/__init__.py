"""The adaptive difficulty engine.

`model` holds the mathematics as pure functions -- no database, no I/O, so it
can be unit tested and simulated directly. `engine` is the thin layer that
reads and writes Postgres around it.
"""
