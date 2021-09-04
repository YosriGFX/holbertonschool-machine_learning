-- 14. In and not out
CREATE TABLE IF NOT EXISTS users (
    id INT NOT NULL AUTO_INCREMENT,
    email VARCHAR(256) NOT NULL UNIQUE,
    name VARCHAR(256),
    country ENUM('US', 'CO', 'TN') DEFAULT 'US',
    PRIMARY KEY (id)
)
