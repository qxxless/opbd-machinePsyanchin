-- Movie recommender database for MySQL 8.0+
-- Run this script in MySQL Workbench using the local MySQL Server connection.

CREATE DATABASE IF NOT EXISTS movie_recommender
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_0900_ai_ci;

USE movie_recommender;

CREATE TABLE IF NOT EXISTS users (
    user_id BIGINT UNSIGNED NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS movies (
    movie_id VARCHAR(10) NOT NULL,
    title VARCHAR(255) NOT NULL,
    release_year SMALLINT UNSIGNED NULL,
    genres VARCHAR(255) NOT NULL DEFAULT '',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (movie_id),
    INDEX idx_movies_title (title),
    INDEX idx_movies_year (release_year)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS ratings (
    user_id BIGINT UNSIGNED NOT NULL,
    movie_id VARCHAR(10) NOT NULL,
    rating DECIMAL(3,1) NOT NULL,
    rating_timestamp BIGINT UNSIGNED NULL,
    source VARCHAR(32) NOT NULL DEFAULT 'MovieTweetings',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, movie_id),
    CONSTRAINT chk_ratings_scale CHECK (rating >= 0 AND rating <= 10),
    CONSTRAINT fk_ratings_user FOREIGN KEY (user_id)
        REFERENCES users (user_id) ON DELETE CASCADE,
    CONSTRAINT fk_ratings_movie FOREIGN KEY (movie_id)
        REFERENCES movies (movie_id) ON DELETE CASCADE,
    INDEX idx_ratings_movie (movie_id),
    INDEX idx_ratings_user_value (user_id, rating)
) ENGINE=InnoDB;

-- After importing the source data, this query should return one row per table.
SELECT 'users' AS table_name, COUNT(*) AS row_count FROM users
UNION ALL
SELECT 'movies', COUNT(*) FROM movies
UNION ALL
SELECT 'ratings', COUNT(*) FROM ratings;
