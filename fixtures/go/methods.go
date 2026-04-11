package methods

import (
	"context"
	"errors"
)

// Repo provides data access methods.
type Repo struct {
	db DB
}

// DB is a database interface.
type DB interface {
	Query(ctx context.Context, query string, args ...any) (string, error)
}

// Find retrieves a record by ID.
func (r *Repo) Find(ctx context.Context, id string) (string, error) {
	return r.db.Query(ctx, "SELECT * WHERE id = ?", id)
}

// Save persists a record.
func (r *Repo) Save(ctx context.Context, data string) error {
	_, err := r.db.Query(ctx, "INSERT INTO records VALUES (?)", data)
	return err
}

// Delete removes a record by ID.
func (r *Repo) Delete(ctx context.Context, id string) error {
	if id == "" {
		return errors.New("id is required")
	}
	_, err := r.db.Query(ctx, "DELETE FROM records WHERE id = ?", id)
	return err
}
