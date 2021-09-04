-- 17. Buy buy buy
CREATE TRIGGER trigger_add AFTER INSERT ON orders FOR EACH ROW UPDATE items SET quantity = quantity - NEW.number WHERE items.name=NEW.item_name;
