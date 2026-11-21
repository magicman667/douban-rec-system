CREATE DATABASE IF NOT EXISTS douban_db;
CREATE EXTERNAL TABLE IF NOT EXISTS douban_db.movies (
  movie_id INT, title STRING, cover_url STRING, year INT, genres ARRAY<STRING>,
  regions ARRAY<STRING>, tags STRING, storyline STRING, main_genre STRING
) STORED AS PARQUET LOCATION '/warehouse/douban_full_v1/movies';
CREATE EXTERNAL TABLE IF NOT EXISTS douban_db.train_ratings (
  rating_id BIGINT, user_id INT, movie_id INT, rating INT, rating_time TIMESTAMP
) STORED AS PARQUET LOCATION '/warehouse/douban_full_v1/train_ratings';
CREATE EXTERNAL TABLE IF NOT EXISTS douban_db.test_ratings (
  rating_id BIGINT, user_id INT, movie_id INT, rating INT, rating_time TIMESTAMP
) STORED AS PARQUET LOCATION '/warehouse/douban_full_v1/test_ratings';
