-- 13. We are all unique!
CREATE TABLE IF NOT EXISTS users (
    id INT NOT NULL AUTO_INCREMENT,
    email VARCHAR(256) NOT NULL UNIQUE,
    name VARCHAR(256),
    PRIMARY KEY (id)
)
