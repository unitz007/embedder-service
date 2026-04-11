package basic

import (
	"fmt"
	"math"
	"time"
)

// User represents a user in the system.
type User struct {
	ID        int       `json:"id" db:"id" yaml:"id"`
	Name      string    `json:"name,omitempty" db:"name" yaml:"name,omitempty"`
	Email     string    `json:"email" db:"email" yaml:"email"`
	CreatedAt time.Time `json:"created_at" db:"created_at" yaml:"created_at"`
}

// Config holds application configuration.
type Config struct {
	Debug bool   `json:"debug" yaml:"debug"`
	Port  int    `json:"port" yaml:"port"`
}

// Reader defines a generic read interface.
type Reader interface {
	Read(p []byte) (n int, err error)
	Close() error
}

func add(a, b int) int {
	return a + b
}

func greet(name string) string {
	return fmt.Sprintf("Hello, %s!", name)
}

func distance(x1, y1, x2, y2 float64) float64 {
	dx := x2 - x1
	dy := y2 - y1
	return math.Sqrt(dx*dx + dy*dy)
}
