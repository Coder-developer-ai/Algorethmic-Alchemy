void setup() {
  pinMode(3, OUTPUT);  
  pinMode(4, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(6, OUTPUT);
  pinMode(7, OUTPUT);
}

void loop() {
  delay(75);           
  digitalWrite(3, HIGH); 
  delay(75);
  digitalWrite(4, HIGH);
  delay(75);
  digitalWrite(5, HIGH);
  delay(75);
  digitalWrite(6, HIGH);
  delay(75);
  digitalWrite(7, HIGH);
  delay(75);
  digitalWrite(7, LOW);
  delay(75);
  digitalWrite(6, LOW);
  delay(75);
  digitalWrite(5, LOW);
  delay(75);
  digitalWrite(4, LOW);
  delay(75);
  digitalWrite(3, LOW);
}
