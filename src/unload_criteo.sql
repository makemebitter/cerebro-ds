CREATE OR REPLACE WRITABLE EXTERNAL TABLE criteo_valid_data_out (LIKE criteo_valid_data) 
   LOCATION ('gpfdist://worker0:8101/valid_0.out',
    'gpfdist://worker1:8101/valid_1.out',
    'gpfdist://worker2:8101/valid_2.out',
    'gpfdist://worker3:8101/valid_3.out',
    'gpfdist://worker4:8101/valid_4.out',
    'gpfdist://worker5:8101/valid_5.out',
    'gpfdist://worker6:8101/valid_6.out',
    'gpfdist://worker7:8101/valid_7.out'
   )
   FORMAT 'TEXT' ( DELIMITER '|' NULL ' ')
   DISTRIBUTED BY (id);
INSERT INTO criteo_valid_data_out SELECT * FROM criteo_valid_data;

CREATE OR REPLACE WRITABLE EXTERNAL TABLE criteo_train_data_out (LIKE criteo_train_data) 
   LOCATION ('gpfdist://worker0:8101/train_0.out',
    'gpfdist://worker1:8101/train_1.out',
    'gpfdist://worker2:8101/train_2.out',
    'gpfdist://worker3:8101/train_3.out',
    'gpfdist://worker4:8101/train_4.out',
    'gpfdist://worker5:8101/train_5.out',
    'gpfdist://worker6:8101/train_6.out',
    'gpfdist://worker7:8101/train_7.out'
   )
   FORMAT 'TEXT' ( DELIMITER '|' NULL ' ')
   DISTRIBUTED BY (id);
INSERT INTO criteo_train_data_out SELECT * FROM criteo_train_data;