let trs = document.querySelectorAll("table tbody tr");
let sql = "";

// începem de la indexul 1 → ignorăm primul rând
for (let i = 1; i < trs.length; i++) {
    let r = trs[i];
    let cells = [...r.querySelectorAll("td")].map(td => td.innerText.trim());
    if (cells.length < 4) continue;

    let sender = cells[1].replace(/"/g, '\\"').replace(/\n/g, ' ');
    let receiver = cells[2].replace(/"/g, '\\"').replace(/\n/g, ' ');
    let message = cells[3].replace(/"/g, '\\"').replace(/\n/g, ' ');
    let sent_date = cells[4];

    sql += `INSERT INTO sms (sender, receiver, message, sent_date) VALUES ("${sender}", "${receiver}", "${message}", "${sent_date}");\n`;
}

// Creăm blob și descărcăm fișierul
let blob = new Blob([sql], {type: "text/sql"});
let a = document.createElement("a");
a.href = URL.createObjectURL(blob);
a.download = "sms.sql";
a.click();
