-- 18. Email validation to sent
DELIMITER $$ CREATE TRIGGER email_check BEFORE UPDATE ON users FOR EACH ROW BEGIN IF STRCMP(OLD.email, NEW.email) <> 0 THEN SET NEW.valid_email = 0;
END IF;
END$$ DELIMITER;
