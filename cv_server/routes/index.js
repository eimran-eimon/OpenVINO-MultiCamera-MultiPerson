var express = require('express');
const path = require('path');
var router = express.Router();
// pg client

const {
	pool
} = require('../config')



/* GET home page. */
router.get('/', async (req, res, ) => {
	try {
		const total_people = await pool.query("select count(*) from (SELECT DISTINCT id FROM track_data WHERE entered='2' GROUP BY id HAVING COUNT(id) > 100) AS temp")
		const people_entered = await pool.query("SELECT count(*) FROM (SELECT DISTINCT id FROM track_data where entered='1' GROUP BY id HAVING COUNT(id) >= 3) AS temp ")
		const people_exit = await pool.query("SELECT count(*) FROM (SELECT DISTINCT id FROM track_data where entered='0' GROUP BY id HAVING COUNT(id) >= 5) AS temp ")
		console.log(total_people.rows[0].count);
		res.render('index', {
			total_people: total_people.rows[0].count,
			people_entered: people_entered.rows[0].count,
			people_exit: people_exit.rows[0].count
		});
	} catch (ex) {
		console.log(ex)
	} finally {
		people_exit.on('end', function () {
			pool.end()
		})

	}
});

// router.get('/', function (req, res, next) {
// 	res.sendFile(path.join(__dirname, '../views/index.html'));
// });

module.exports = router;